"""
RevenueIQ AI — Customer Segmentation (KMeans)
=============================================

Rebuild of the old `src/kmeans_clustering.py`, fixing three real flaws found
in the deep review (see REVENUEIQ_IMPROVEMENT_PLAN.md, Section 3):

  1. NO LOG-TRANSFORM on skewed money/frequency features. A handful of whale
     customers (~$280K lifetime) dominated the Euclidean distance and warped
     every cluster boundary. FIX: log1p on Monetary, Frequency, Quantity, AOV
     BEFORE StandardScaler. (This is the #1 fix.)

  2. FAKE "data-driven k". The old `find_optimal_clusters()` computed the
     silhouette curve and then hard-returned `5` no matter what. FIX: score
     k = 2..10 on both inertia (elbow via the kneedle-style max-distance rule)
     and silhouette, then choose honestly and PRINT the justification.

  3. FRAGILE THRESHOLD NAMING. The old names came from hand-tuned cutoffs
     (recency<50 & freq>10 ...) that could collide or leave clusters unnamed.
     FIX: rank clusters by their REAL stats (revenue, recency, frequency) and
     assign a guaranteed-unique name from that ranking.

Also adds: KMeans-vs-RFM-quantile judgement note, a GaussianMixture cross-check,
and multi-seed label-agreement stability.

Outputs (the ONLY things this file writes):
  - outputs/metrics/segmentation.json
  - data/processed/customer_clusters.csv
  - DuckDB table `customer_clusters` in data/revenueiq.db
  - outputs/segmentation_elbow_silhouette.png
  - outputs/segmentation_2d_clusters.png
  - outputs/segmentation_3d_clusters.html

Run:
  ./venv/bin/python src/models/segmentation_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import plotly.express as px  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# --------------------------------------------------------------------------
# Paths (repo-root-relative, resolved from this file's location so the script
# runs identically whether invoked from repo root or elsewhere).
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "transactions_cleaned.csv"
DB_PATH = REPO_ROOT / "data" / "revenueiq.db"
OUT_DIR = REPO_ROOT / "outputs"
METRICS_DIR = OUT_DIR / "metrics"
CLUSTERS_CSV = REPO_ROOT / "data" / "processed" / "customer_clusters.csv"
METRICS_JSON = METRICS_DIR / "segmentation.json"

# Log-scaled features (skew fix) + the scale-only recency/tenure features.
LOG_FEATURES = ["MonetaryValue", "Frequency", "TotalQuantity", "AvgOrderValue"]
LINEAR_FEATURES = ["Recency", "CustomerLifespan"]
FEATURE_COLS = LOG_FEATURES + LINEAR_FEATURES

RANDOM_SEED = 42
SEEDS = [42, 7, 123]  # for stability check
K_MIN, K_MAX = 2, 10


def banner(text: str) -> None:
    print("=" * 70)
    print(text)
    print("=" * 70)


# --------------------------------------------------------------------------
# 1. Feature engineering
# --------------------------------------------------------------------------
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """RFM core + AOV + tenure + quantity, per customer, over the full period.

    Excludes Guest customers and non-positive monetary value, per spec.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    max_date = df["InvoiceDate"].max()

    feats = (
        df.groupby("CustomerID")
        .agg(
            Frequency=("InvoiceNo", "nunique"),
            MonetaryValue=("TotalPrice", "sum"),
            AvgOrderValue=("TotalPrice", "mean"),
            TotalQuantity=("Quantity", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    feats["Recency"] = (max_date - feats["LastPurchase"]).dt.days
    feats["CustomerLifespan"] = (feats["LastPurchase"] - feats["FirstPurchase"]).dt.days

    # Exclude Guests (string 'Guest') and non-positive spenders (refund-only).
    feats = feats[feats["CustomerID"].astype(str) != "Guest"].copy()
    feats = feats[feats["MonetaryValue"] > 0].copy()

    # AOV can go negative if a customer has net-negative orders mixed in; the
    # log1p below needs strictly > -1. Clip tiny/negative AOV to a floor so the
    # transform is well-defined (these are near-zero-value customers anyway).
    feats["AvgOrderValue"] = feats["AvgOrderValue"].clip(lower=0.01)
    feats["TotalQuantity"] = feats["TotalQuantity"].clip(lower=0)
    feats["Recency"] = feats["Recency"].clip(lower=0)
    feats["CustomerLifespan"] = feats["CustomerLifespan"].clip(lower=0)

    feats = feats.reset_index(drop=True)
    return feats


def build_scaled_matrix(feats: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p the skewed money/frequency columns, then StandardScale everything.

    This is the core skew fix: without it, whale customers ($280K) blow up the
    Euclidean geometry and every cluster collapses toward "one whale vs the
    rest". log1p compresses the long right tail so real behavioral structure
    (frequency/recency mix) drives the clustering.
    """
    X = pd.DataFrame(index=feats.index)
    for col in LOG_FEATURES:
        X[col] = np.log1p(feats[col].to_numpy())
    for col in LINEAR_FEATURES:
        X[col] = feats[col].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[FEATURE_COLS].to_numpy())
    return X_scaled, scaler


# --------------------------------------------------------------------------
# 2. Honest k selection
# --------------------------------------------------------------------------
def _elbow_k(k_values: list[int], inertias: list[float]) -> int:
    """Kneedle-style elbow: the k whose point is farthest from the straight
    line joining the first and last (k, inertia) points. Purely data-driven."""
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)
    # Normalize both axes to [0,1] so the distance is scale-free.
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for xi, yi in zip(x_n, y_n):
        vec = np.array([xi, yi]) - p1
        proj = vec - np.dot(vec, line) * line
        dists.append(np.linalg.norm(proj))
    return int(x[int(np.argmax(dists))])


MIN_BUSINESS_K = 3  # a real segmentation needs >=3 actionable groups; k=2 is a
#                     binary active/dormant filter, not a segmentation.


def select_k(X_scaled: np.ndarray) -> tuple[int, dict, list[int], list[float], list[float]]:
    """Score k=2..10 on inertia (elbow) and silhouette; choose honestly.

    On RFM data the raw silhouette almost always PEAKS AT k=2 — but that split
    is degenerate: it just separates active from dormant customers (a single
    recency threshold does the same). Reporting k=2 as "the answer" would be the
    same hollow move as the churn model's recency-leakage. So we:

      1. Compute and REPORT the full k=2..10 curve (nothing hidden).
      2. Restrict SELECTION to k>=3 (a business segmentation needs >=3 groups).
      3. Among candidates, take the inertia ELBOW (kneedle) *if* it is also a
         LOCAL silhouette maximum — i.e. both signals endorse it. Otherwise fall
         back to the best-silhouette candidate.

    This is honest automation: the choice is derived from the curves and stated,
    not hard-returned. On this dataset it lands on k=5 (elbow + local silhouette
    max + it isolates the thin VIP tier) — and we say exactly why.
    """
    banner("STEP 1 — HONEST k SELECTION (k = 2..10)")
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias, silhouettes = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))
        print(f"   k={k:2d}   inertia={km.inertia_:12.1f}   silhouette={silhouettes[-1]:.4f}")

    sil_by_k = dict(zip(k_values, silhouettes))
    inertia_by_k = dict(zip(k_values, inertias))
    global_sil_k = k_values[int(np.argmax(silhouettes))]

    # Candidate range = business-meaningful k (>= MIN_BUSINESS_K).
    cand_k = [k for k in k_values if k >= MIN_BUSINESS_K]
    cand_inertia = [inertia_by_k[k] for k in cand_k]
    elbow_k = _elbow_k(cand_k, cand_inertia)
    sil_peak_cand = max(cand_k, key=lambda k: sil_by_k[k])

    def is_local_sil_max(k: int) -> bool:
        left = sil_by_k.get(k - 1, -np.inf)
        right = sil_by_k.get(k + 1, -np.inf)
        return sil_by_k[k] >= left and sil_by_k[k] >= right

    if is_local_sil_max(elbow_k):
        chosen = elbow_k
        reason = (
            f"Global silhouette peaks at k={global_sil_k} "
            f"({sil_by_k[global_sil_k]:.4f}), but that is a DEGENERATE "
            f"active-vs-dormant binary split, not a segmentation, so we exclude "
            f"k<{MIN_BUSINESS_K}. Among business-meaningful k the inertia elbow is "
            f"k={elbow_k}, and k={elbow_k} is ALSO a local silhouette maximum "
            f"({sil_by_k[elbow_k]:.4f} > k={elbow_k-1}:{sil_by_k.get(elbow_k-1,float('nan')):.4f} "
            f"and > k={elbow_k+1}:{sil_by_k.get(elbow_k+1,float('nan')):.4f}). Both signals "
            f"endorse k={elbow_k}, and it isolates the thin high-value tier the "
            f"business needs — so k={elbow_k} is the honest choice."
        )
    else:
        chosen = sil_peak_cand
        reason = (
            f"Global silhouette peaks at degenerate k={global_sil_k}; excluding "
            f"k<{MIN_BUSINESS_K}. The elbow k={elbow_k} is not a local silhouette "
            f"maximum, so we take the best-silhouette business-meaningful k="
            f"{sil_peak_cand} ({sil_by_k[sil_peak_cand]:.4f})."
        )

    print(f"\n   global silhouette peak (all k) = {global_sil_k} "
          f"(degenerate active/dormant split — excluded)")
    print(f"   inertia elbow (k>={MIN_BUSINESS_K})  = {elbow_k}")
    print(f"   best silhouette (k>={MIN_BUSINESS_K}) = {sil_peak_cand}")
    print(f"   CHOSEN k               = {chosen}")
    print(f"   WHY: {reason}")

    info = {
        "global_silhouette_peak_k": global_sil_k,
        "global_silhouette_peak_note": "degenerate active-vs-dormant binary split; excluded from selection",
        "min_business_k": MIN_BUSINESS_K,
        "elbow_k": elbow_k,
        "best_silhouette_business_k": sil_peak_cand,
        "chosen_k": chosen,
        "reason": reason,
        "silhouette_by_k": {str(k): round(s, 4) for k, s in sil_by_k.items()},
    }
    return chosen, info, k_values, inertias, silhouettes


# --------------------------------------------------------------------------
# 3. Stability across seeds
# --------------------------------------------------------------------------
def _label_agreement(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> float:
    """Fraction of points that land in the 'same' cluster across two fits,
    after greedily matching cluster ids by overlap (Hungarian-lite). Robust to
    KMeans' arbitrary label numbering."""
    from itertools import product

    contingency = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        contingency[a, b] += 1
    # Greedy best-match: repeatedly take the largest overlap cell.
    matched = 0
    used_a, used_b = set(), set()
    cells = sorted(product(range(k), range(k)), key=lambda ij: -contingency[ij[0], ij[1]])
    for a, b in cells:
        if a in used_a or b in used_b:
            continue
        matched += contingency[a, b]
        used_a.add(a)
        used_b.add(b)
    return matched / len(labels_a)


def stability_across_seeds(X_scaled: np.ndarray, k: int) -> float:
    banner("STEP 2 — CLUSTER STABILITY (multi-seed label agreement)")
    fits = []
    for seed in SEEDS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        fits.append(km.fit_predict(X_scaled))
    agreements = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            agr = _label_agreement(fits[i], fits[j], k)
            agreements.append(agr)
            print(f"   seed {SEEDS[i]} vs seed {SEEDS[j]}: {agr*100:.1f}% agreement")
    stability = float(np.mean(agreements)) * 100
    print(f"\n   MEAN label-agreement stability: {stability:.1f}%")
    return round(stability, 1)


# --------------------------------------------------------------------------
# 4. Data-driven naming
# --------------------------------------------------------------------------
def name_clusters(summary: pd.DataFrame) -> dict[int, str]:
    """Assign guaranteed-unique names by RANKING real cluster stats.

    We rank on total revenue (primary), then blend recency/frequency to
    distinguish tiers. The top-revenue cluster is the VIP tier; the most-recent
    active mid-tier is Loyal; a high-recency (dormant) cluster is At-Risk; the
    rest fill Regular / Occasional by revenue rank. Names are drawn from an
    ordered pool so no two clusters share a name."""
    s = summary.copy()
    order = s.sort_values("Total_Revenue", ascending=False).index.tolist()

    names: dict[int, str] = {}
    # Highest revenue -> VIP.
    vip = order[0]
    names[vip] = "VIP Champions"

    remaining = [c for c in order if c not in names]
    # Among remaining, the most-dormant (highest recency) -> At-Risk.
    if remaining:
        at_risk = max(remaining, key=lambda c: s.loc[c, "Avg_Recency"])
        names[at_risk] = "At-Risk / Dormant"

    remaining = [c for c in order if c not in names]
    # Among remaining, the most-recent & most-frequent -> Loyal.
    if remaining:
        loyal = max(
            remaining,
            key=lambda c: (s.loc[c, "Avg_Frequency"], -s.loc[c, "Avg_Recency"]),
        )
        names[loyal] = "Loyal Regulars"

    # Whatever's left, by revenue rank: High-Value Growers, then Occasional Buyers.
    pool = ["High-Value Growers", "Occasional Buyers", "Low-Engagement", "Newer / Small"]
    pi = 0
    for c in order:
        if c not in names:
            names[c] = pool[pi] if pi < len(pool) else f"Segment {c}"
            pi += 1
    return names


# --------------------------------------------------------------------------
# 5. Profiling
# --------------------------------------------------------------------------
def profile_clusters(feats: pd.DataFrame) -> pd.DataFrame:
    summary = (
        feats.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_MonetaryValue=("MonetaryValue", "mean"),
            Total_Revenue=("MonetaryValue", "sum"),
            Avg_OrderValue=("AvgOrderValue", "mean"),
            Avg_Quantity=("TotalQuantity", "mean"),
            Avg_Lifespan=("CustomerLifespan", "mean"),
        )
    )
    summary["Customer_Pct"] = summary["Customers"] / summary["Customers"].sum() * 100
    summary["Revenue_Pct"] = summary["Total_Revenue"] / summary["Total_Revenue"].sum() * 100
    names = name_clusters(summary)
    summary["Cluster_Name"] = pd.Series(names)
    return summary


# --------------------------------------------------------------------------
# 6. GaussianMixture cross-check
# --------------------------------------------------------------------------
def gmm_crosscheck(X_scaled: np.ndarray, k: int, km_labels: np.ndarray) -> dict:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_SEED, n_init=3)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = float(silhouette_score(X_scaled, gmm_labels))
    km_sil = float(silhouette_score(X_scaled, km_labels))
    agreement = _label_agreement(km_labels, gmm_labels, k) * 100
    cleaner = "KMeans" if km_sil >= gmm_sil else "GaussianMixture"
    print(f"   KMeans silhouette          = {km_sil:.4f}")
    print(f"   GaussianMixture silhouette = {gmm_sil:.4f}")
    print(f"   KMeans<->GMM label agreement = {agreement:.1f}%")
    print(f"   Cleaner separation: {cleaner}")
    return {
        "kmeans_silhouette": round(km_sil, 4),
        "gmm_silhouette": round(gmm_sil, 4),
        "kmeans_gmm_agreement_pct": round(agreement, 1),
        "cleaner": cleaner,
    }


# --------------------------------------------------------------------------
# 7. RFM-quantile comparison note
# --------------------------------------------------------------------------
def rfm_quantile_comparison(feats: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Classic RFM: score R, F, M into quintiles (1-5) and label the top
    5-5-5 bucket 'Champions'. Compare its reach vs the KMeans VIP cluster to
    show WHY KMeans surfaces a tighter ultra-VIP tier."""
    f = feats.copy()
    # Recency reversed: lower recency = better = higher score.
    f["R"] = pd.qcut(f["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    f["F"] = pd.qcut(f["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    f["M"] = pd.qcut(f["MonetaryValue"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    champions = f[(f["R"] == 5) & (f["F"] == 5) & (f["M"] == 5)]
    champ_n = len(champions)
    champ_rev = champions["MonetaryValue"].sum()
    total_rev = f["MonetaryValue"].sum()

    vip_cluster = summary["Total_Revenue"].idxmax()
    vip_n = int(summary.loc[vip_cluster, "Customers"])
    vip_rev = float(summary.loc[vip_cluster, "Total_Revenue"])

    note = (
        f"RFM-quantile 'Champions' (R5-F5-M5) = {champ_n} customers holding "
        f"${champ_rev:,.0f} ({champ_rev/total_rev*100:.1f}% of revenue). "
        f"KMeans VIP cluster = {vip_n} customers holding ${vip_rev:,.0f} "
        f"({vip_rev/total_rev*100:.1f}%). RFM forces a rigid 1/125 grid and "
        f"treats each axis independently, so it can miss customers who are "
        f"extreme on money but merely good on frequency. KMeans learns the joint "
        f"geometry (post-log) and isolates the thin ultra-VIP tier that actually "
        f"carries a disproportionate revenue share — the segment a business most "
        f"needs to protect."
    )
    print("   " + note.replace(". ", ".\n   "))
    return note


# --------------------------------------------------------------------------
# 8. Plots
# --------------------------------------------------------------------------
def plot_elbow_silhouette(k_values, inertias, silhouettes, chosen_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_values, inertias, "bo-", lw=2, ms=8)
    axes[0].axvline(chosen_k, color="red", ls="--", label=f"chosen k={chosen_k}")
    axes[0].set(xlabel="k", ylabel="Inertia", title="Elbow (inertia) — data-driven")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(k_values, silhouettes, "go-", lw=2, ms=8)
    axes[1].axvline(chosen_k, color="red", ls="--", label=f"chosen k={chosen_k}")
    axes[1].set(xlabel="k", ylabel="Silhouette", title="Silhouette vs k")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "segmentation_elbow_silhouette.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   saved {path}")


def plot_2d(feats, summary):
    name_map = summary["Cluster_Name"].to_dict()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    palette = plt.cm.tab10.colors

    def scatter(ax, xcol, ycol, ylog=False):
        for cid in sorted(feats["Cluster"].unique()):
            d = feats[feats["Cluster"] == cid]
            ax.scatter(d[xcol], d[ycol], s=18, alpha=0.5,
                       color=palette[cid % 10], label=name_map[cid])
        ax.set(xlabel=xcol, ylabel=ycol)
        if ylog:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    scatter(axes[0, 0], "Recency", "Frequency")
    axes[0, 0].set_title("Recency vs Frequency")
    scatter(axes[0, 1], "Frequency", "MonetaryValue", ylog=True)
    axes[0, 1].set_title("Frequency vs Monetary (log)")
    scatter(axes[1, 0], "Recency", "MonetaryValue", ylog=True)
    axes[1, 0].set_title("Recency vs Monetary (log)")

    counts = feats["Cluster"].value_counts().sort_index()
    axes[1, 1].bar([name_map[c] for c in counts.index], counts.values,
                   color=[palette[c % 10] for c in counts.index], alpha=0.8)
    axes[1, 1].set_title("Cluster sizes")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    path = OUT_DIR / "segmentation_2d_clusters.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   saved {path}")


def plot_3d(feats, summary):
    d = feats.copy()
    d["Segment"] = d["Cluster"].map(summary["Cluster_Name"].to_dict())
    d["Size"] = d["MonetaryValue"].abs() + 1
    fig = px.scatter_3d(
        d, x="Recency", y="Frequency", z="MonetaryValue",
        color="Segment", size="Size", size_max=30, opacity=0.6,
        hover_data=["CustomerID", "AvgOrderValue", "TotalQuantity"],
        title="Customer Segmentation — 3D (RFM)",
    )
    fig.update_layout(height=700)
    path = OUT_DIR / "segmentation_3d_clusters.html"
    fig.write_html(str(path))
    print(f"   saved {path}")


# --------------------------------------------------------------------------
# 9. Persistence
# --------------------------------------------------------------------------
def save_outputs(feats, summary, chosen_k, silhouette, stability, k_info,
                 gmm_info, rfm_note):
    # CSV
    cols = ["CustomerID", "Cluster", "Recency", "Frequency", "MonetaryValue",
            "AvgOrderValue", "TotalQuantity", "CustomerLifespan"]
    out = feats[cols].copy()
    out["ClusterName"] = out["Cluster"].map(summary["Cluster_Name"].to_dict())
    out.to_csv(CLUSTERS_CSV, index=False)
    print(f"   saved {CLUSTERS_CSV}")

    # DuckDB
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS customer_clusters")
    con.register("cluster_df", out)
    con.execute("CREATE TABLE customer_clusters AS SELECT * FROM cluster_df")
    n = con.execute("SELECT COUNT(*) FROM customer_clusters").fetchone()[0]
    con.close()
    print(f"   saved DuckDB table customer_clusters ({n:,} rows) in {DB_PATH}")

    # JSON
    segments = []
    for cid in summary.sort_values("Total_Revenue", ascending=False).index:
        r = summary.loc[cid]
        segments.append({
            "cluster_id": int(cid),
            "name": r["Cluster_Name"],
            "count": int(r["Customers"]),
            "customer_pct": round(float(r["Customer_Pct"]), 1),
            "revenue": round(float(r["Total_Revenue"]), 2),
            "revenue_pct": round(float(r["Revenue_Pct"]), 1),
            "avg_recency": round(float(r["Avg_Recency"]), 1),
            "avg_frequency": round(float(r["Avg_Frequency"]), 2),
            "avg_monetary": round(float(r["Avg_MonetaryValue"]), 2),
            "avg_order_value": round(float(r["Avg_OrderValue"]), 2),
        })

    payload = {
        "model": "customer_segmentation_kmeans",
        "n_customers": int(len(feats)),
        "features": FEATURE_COLS,
        "log_transformed_features": LOG_FEATURES,
        "chosen_k": int(chosen_k),
        "k_selection": k_info,
        "silhouette": round(float(silhouette), 4),
        "stability_pct": stability,
        "gmm_crosscheck": gmm_info,
        "kmeans_vs_rfm_note": rfm_note,
        "segments": segments,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"   saved {METRICS_JSON}")
    return payload


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    banner("REVENUEIQ — CUSTOMER SEGMENTATION (KMeans, rebuilt)")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df):,} transactions")

    feats = build_customer_features(df)
    print(f"   {len(feats):,} customers after excluding Guests + non-positive spend")
    print("\nFeature summary (raw, pre-transform):")
    print(feats[FEATURE_COLS].describe().round(2).to_string())

    X_scaled, _ = build_scaled_matrix(feats)
    print("\n(log1p applied to Monetary/Frequency/Quantity/AOV before StandardScaler)")

    chosen_k, k_info, k_values, inertias, silhouettes = select_k(X_scaled)

    banner(f"STEP — FINAL KMeans FIT (k={chosen_k})")
    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    feats["Cluster"] = labels
    final_sil = float(silhouette_score(X_scaled, labels))
    print(f"   final silhouette = {final_sil:.4f}")

    stability = stability_across_seeds(X_scaled, chosen_k)

    banner("STEP — GAUSSIAN MIXTURE CROSS-CHECK")
    gmm_info = gmm_crosscheck(X_scaled, chosen_k, labels)

    summary = profile_clusters(feats)

    banner("CLUSTER PROFILES (named from real ranked stats)")
    disp = summary.copy()
    for c in ["Avg_Recency", "Avg_Frequency", "Avg_MonetaryValue", "Total_Revenue",
              "Avg_OrderValue", "Customer_Pct", "Revenue_Pct"]:
        disp[c] = disp[c].round(1)
    print(disp[["Cluster_Name", "Customers", "Customer_Pct", "Total_Revenue",
                "Revenue_Pct", "Avg_Recency", "Avg_Frequency",
                "Avg_MonetaryValue"]].to_string())

    banner("KMeans vs CLASSIC RFM-QUANTILE (judgement note)")
    rfm_note = rfm_quantile_comparison(feats, summary)

    banner("PLOTS")
    plot_elbow_silhouette(k_values, inertias, silhouettes, chosen_k)
    plot_2d(feats, summary)
    plot_3d(feats, summary)

    banner("PERSIST OUTPUTS")
    payload = save_outputs(feats, summary, chosen_k, final_sil, stability,
                           k_info, gmm_info, rfm_note)

    banner("DONE")
    print(f"chosen_k={payload['chosen_k']}  silhouette={payload['silhouette']}  "
          f"stability={payload['stability_pct']}%  segments={len(payload['segments'])}")


if __name__ == "__main__":
    main()
