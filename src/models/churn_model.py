"""
RevenueIQ AI - Churn Model (rebuild, Agent A)
=============================================

Fixes the target-leakage bug in `src/predictive_models.py::predict_customer_churn()`:
the old model defined `Churned = Recency > 90` and then fed `Recency` itself in as a
feature, so it was predicting the label it was handed (fake ~95% F1 / near-1.0 AUC).

This version uses a **time-split design**:
  - Observation window: first ~10 months of the dataset (features built ONLY from here).
  - Outcome window: remaining ~last months of the dataset.
  - Label: churned = 1 if a customer who was active in the observation window makes
    NO purchase in the outcome window. Customers absent from the observation window
    are excluded entirely (they aren't "active customers" we could have scored).
  - Recency-to-observation-end (i.e. days between a customer's last purchase and the
    cutoff date) is NEVER used as a feature — that quantity is mechanically almost
    identical to the label definition and is exactly what leaked before.

Run:
    cd /Users/nagapremsaipendela/Dev/Revenueiq-ai
    ./venv/bin/python src/models/churn_model.py

Writes:
    outputs/metrics/churn.json
    data/processed/at_risk_customers.csv
    outputs/shap/churn_shap_summary.png
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "transactions_sales_only.csv")
METRICS_OUT = os.path.join(REPO_ROOT, "outputs", "metrics", "churn.json")
AT_RISK_OUT = os.path.join(REPO_ROOT, "data", "processed", "at_risk_customers.csv")
SHAP_PNG_OUT = os.path.join(REPO_ROOT, "outputs", "shap", "churn_shap_summary.png")

# Cutoff between observation window and outcome window (per improvement plan Sec. 2).
CUTOFF = pd.Timestamp("2011-09-30")
RECENT_WINDOW_DAYS = 30  # for the spend-trend feature (last 30d vs prior 30d, inside obs window)
RANDOM_STATE = 42
TOP_N_AT_RISK_PRECISION = 200

FEATURE_COLS = [
    "frequency",
    "monetary",
    "aov",
    "avg_gap_days",
    "std_gap_days",
    "tenure_days",
    "distinct_products",
    "total_quantity",
    "spend_trend_30d",
]


def load_data(path: str) -> pd.DataFrame:
    print("Loading transactions...")
    usecols = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice",
               "CustomerID", "TotalPrice"]
    df = pd.read_csv(path, usecols=usecols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Exclude Guests / null customers — customer-level modeling needs a real ID.
    df = df[df["CustomerID"].notna()]
    df = df[df["CustomerID"].astype(str) != "Guest"]
    # CustomerIDs are floats like 17850.0 in this file — normalize to int.
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)

    print(f"Loaded {len(df):,} transaction rows, {df['CustomerID'].nunique():,} unique customers "
          f"(Guests/nulls excluded).")
    print(f"Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    return df


def build_features(window_df: pd.DataFrame, window_end: pd.Timestamp) -> pd.DataFrame:
    """Build customer-level features using ONLY rows inside `window_df`.

    `window_end` is used purely to define the internal 30d/30d spend-trend split —
    it is NOT used to compute a recency-to-window-end feature.
    """
    g = window_df.groupby("CustomerID")

    base = g.agg(
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
        total_quantity=("Quantity", "sum"),
        distinct_products=("StockCode", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    base["aov"] = base["monetary"] / base["frequency"]
    base["tenure_days"] = (base["last_purchase"] - base["first_purchase"]).dt.days

    # Inter-purchase gap stats (days between distinct purchase days within the window).
    def gap_stats(dates: pd.Series) -> pd.Series:
        days = np.sort(dates.dt.normalize().unique())
        if len(days) < 2:
            return pd.Series({"avg_gap_days": np.nan, "std_gap_days": 0.0})
        diffs = np.diff(days) / np.timedelta64(1, "D")
        return pd.Series({
            "avg_gap_days": float(diffs.mean()),
            "std_gap_days": float(diffs.std(ddof=0)) if len(diffs) > 1 else 0.0,
        })

    gaps = g["InvoiceDate"].apply(gap_stats).unstack()
    base = base.join(gaps)

    # Single-purchase customers get no gap signal — fill with the observation window
    # length itself (a large, sentinel-like value meaning "never repurchased in-window"),
    # NOT anything derived from the cutoff/label boundary.
    window_length_days = (window_df["InvoiceDate"].max() - window_df["InvoiceDate"].min()).days
    base["avg_gap_days"] = base["avg_gap_days"].fillna(window_length_days)
    base["std_gap_days"] = base["std_gap_days"].fillna(0.0)

    # Spend trend: last 30d of the OBSERVATION window vs the prior 30d — both entirely
    # inside the observation window, so this does not leak outcome-window information.
    recent_start = window_end - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    prior_start = window_end - pd.Timedelta(days=2 * RECENT_WINDOW_DAYS)

    recent_spend = (
        window_df[window_df["InvoiceDate"] > recent_start]
        .groupby("CustomerID")["TotalPrice"].sum()
    )
    prior_spend = (
        window_df[(window_df["InvoiceDate"] > prior_start) & (window_df["InvoiceDate"] <= recent_start)]
        .groupby("CustomerID")["TotalPrice"].sum()
    )
    trend = (recent_spend.reindex(base.index, fill_value=0.0)
              - prior_spend.reindex(base.index, fill_value=0.0))
    base["spend_trend_30d"] = trend

    base = base.drop(columns=["first_purchase", "last_purchase"])
    return base


def leakage_self_check(feature_cols):
    print("\n--- Leakage self-check ---")
    forbidden_markers = ["recency", "days_since_last", "days_to_cutoff", "days_since_cutoff"]
    hits = [c for c in feature_cols if any(m in c.lower() for m in forbidden_markers)]
    print(f"Feature columns used: {feature_cols}")
    assert len(hits) == 0, f"LEAKAGE: forbidden recency-to-cutoff-like feature(s) found: {hits}"
    print("PASSED — no recency-to-cutoff feature present in the feature set.")
    print("(tenure_days and avg_gap_days are computed entirely inside the observation "
          "window and do not reference the cutoff/label boundary.)")


def precision_at_k(y_true, y_score, k):
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(np.mean(np.asarray(y_true)[order]))


def main():
    df = load_data(DATA_PATH)

    data_start = df["InvoiceDate"].min()
    data_end = df["InvoiceDate"].max()
    print(f"\nObservation window: {data_start.date()} -> {CUTOFF.date()}  "
          f"({(CUTOFF - data_start).days} days, ~{(CUTOFF - data_start).days/30.4:.1f} months)")
    print(f"Outcome window:     {CUTOFF.date()} -> {data_end.date()}  "
          f"({(data_end - CUTOFF).days} days, ~{(data_end - CUTOFF).days/30.4:.1f} months)")

    obs_df = df[df["InvoiceDate"] <= CUTOFF]
    outcome_df = df[df["InvoiceDate"] > CUTOFF]

    active_obs_customers = set(obs_df["CustomerID"].unique())
    purchased_in_outcome = set(outcome_df["CustomerID"].unique())

    print(f"\nCustomers active in observation window: {len(active_obs_customers):,}")
    print(f"Of those, purchased again in outcome window: "
          f"{len(active_obs_customers & purchased_in_outcome):,}")

    print("\nBuilding observation-window features (this is the ONLY data the model sees)...")
    features = build_features(obs_df, CUTOFF)
    # Restrict strictly to customers active in observation (excludes any accidental extras).
    features = features.loc[features.index.isin(active_obs_customers)]

    features["churned"] = (~features.index.isin(purchased_in_outcome)).astype(int)

    n_customers = len(features)
    churn_rate = float(features["churned"].mean())
    print(f"\nModeling population: {n_customers:,} customers")
    print(f"Churn rate (no purchase in outcome window): {churn_rate*100:.2f}%")
    print(f"Class balance -> active: {(features['churned']==0).sum():,}, "
          f"churned: {(features['churned']==1).sum():,}")

    leakage_self_check(FEATURE_COLS)

    X = features[FEATURE_COLS].fillna(0)
    y = features["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} customers | Test: {len(X_test):,} customers")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    # 1) Logistic Regression baseline
    print("\n--- Model 1: Logistic Regression (baseline) ---")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    proba = lr.predict_proba(X_test_s)[:, 1]
    results["logistic_regression"] = {
        "auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision_at_200": precision_at_k(y_test.values, proba, TOP_N_AT_RISK_PRECISION),
    }
    fitted_models["logistic_regression"] = (lr, proba)
    print(f"  AUC={results['logistic_regression']['auc']:.4f}  "
          f"PR-AUC={results['logistic_regression']['pr_auc']:.4f}  "
          f"P@200={results['logistic_regression']['precision_at_200']:.4f}")

    # 2) Random Forest
    print("\n--- Model 2: Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    proba = rf.predict_proba(X_test_s)[:, 1]
    results["random_forest"] = {
        "auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision_at_200": precision_at_k(y_test.values, proba, TOP_N_AT_RISK_PRECISION),
    }
    fitted_models["random_forest"] = (rf, proba)
    print(f"  AUC={results['random_forest']['auc']:.4f}  "
          f"PR-AUC={results['random_forest']['pr_auc']:.4f}  "
          f"P@200={results['random_forest']['precision_at_200']:.4f}")

    # 3) XGBoost
    print("\n--- Model 3: XGBoost ---")
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_model.fit(X_train_s, y_train)
    proba = xgb_model.predict_proba(X_test_s)[:, 1]
    results["xgboost"] = {
        "auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision_at_200": precision_at_k(y_test.values, proba, TOP_N_AT_RISK_PRECISION),
    }
    fitted_models["xgboost"] = (xgb_model, proba)
    print(f"  AUC={results['xgboost']['auc']:.4f}  "
          f"PR-AUC={results['xgboost']['pr_auc']:.4f}  "
          f"P@200={results['xgboost']['precision_at_200']:.4f}")

    # Select best model by PR-AUC (primary metric for an imbalanced problem), tie-break on AUC.
    best_name = max(results, key=lambda k: (results[k]["pr_auc"], results[k]["auc"]))
    best_model, best_proba = fitted_models[best_name]
    print(f"\nBest model by PR-AUC: {best_name}")

    cm = confusion_matrix(y_test, (best_proba >= 0.5).astype(int))
    print(f"Confusion matrix (threshold=0.5) for {best_name}:\n{cm}")

    print("\nSanity gate check: ROC-AUC should land ~0.75-0.90 (not ~1.0).")
    for name, r in results.items():
        flag = "  <-- SUSPICIOUS (near-perfect, check for leakage)" if r["auc"] > 0.97 else ""
        print(f"  {name:20s} AUC={r['auc']:.4f}{flag}")

    # ---- SHAP explainability on the best model ----
    print(f"\nComputing SHAP values for {best_name}...")
    X_test_df = pd.DataFrame(X_test_s, columns=FEATURE_COLS)
    if best_name == "logistic_regression":
        explainer = shap.LinearExplainer(best_model, pd.DataFrame(X_train_s, columns=FEATURE_COLS))
        shap_values = explainer.shap_values(X_test_df)
    else:
        explainer = shap.TreeExplainer(best_model)
        raw_shap = explainer.shap_values(X_test_df)
        # Some sklearn/xgboost versions return a list per-class for binary classifiers.
        shap_values = raw_shap[1] if isinstance(raw_shap, list) else raw_shap

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = (
        pd.Series(mean_abs_shap, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    print("\nTop SHAP feature importances (mean |SHAP value| on test set):")
    for feat, val in shap_importance.items():
        print(f"  {feat:20s} {val:.4f}")

    top_shap_features = [
        {"feature": feat, "mean_abs_shap": float(val)}
        for feat, val in shap_importance.items()
    ]

    os.makedirs(os.path.dirname(SHAP_PNG_OUT), exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title(f"Churn risk drivers — SHAP summary ({best_name})\n"
              "(attribution, not proven causation)")
    plt.tight_layout()
    plt.savefig(SHAP_PNG_OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {SHAP_PNG_OUT}")

    # ---- Refit best model type on the FULL labeled population for the actionable list ----
    # (Reported metrics above come only from the held-out test split; this refit is a
    # separate, standard "score the known customer base with the validated model" step
    # for the business-facing at-risk CSV, not part of the evaluation.)
    print(f"\nRefitting {best_name} on the full observation-window population "
          f"({n_customers:,} customers) for the at-risk export...")
    X_full_s = scaler.fit_transform(X)
    if best_name == "logistic_regression":
        final_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif best_name == "random_forest":
        final_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        n_pos_full = int(y.sum())
        n_neg_full = int((y == 0).sum())
        final_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=n_neg_full / max(n_pos_full, 1), eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    final_model.fit(X_full_s, y)
    full_proba = final_model.predict_proba(X_full_s)[:, 1]

    at_risk = pd.DataFrame({
        "CustomerID": features.index,
        "churn_prob": full_proba,
        "revenue_at_risk": full_proba * features["monetary"].values,
    }).sort_values("churn_prob", ascending=False)

    os.makedirs(os.path.dirname(AT_RISK_OUT), exist_ok=True)
    at_risk.to_csv(AT_RISK_OUT, index=False)
    print(f"Saved at-risk customer list -> {AT_RISK_OUT} ({len(at_risk):,} rows)")
    print(f"Total revenue_at_risk (sum of churn_prob * monetary across all scored customers): "
          f"${at_risk['revenue_at_risk'].sum():,.2f}")

    # ---- Write metrics JSON ----
    metrics = {
        "auc": results[best_name]["auc"],
        "pr_auc": results[best_name]["pr_auc"],
        "precision_at_200": results[best_name]["precision_at_200"],
        "churn_rate": churn_rate,
        "n_customers": n_customers,
        "best_model": best_name,
        "top_shap_features": top_shap_features,
        "all_models_compared": results,
        "observation_window": {"start": str(data_start.date()), "end": str(CUTOFF.date())},
        "outcome_window": {"start": str(CUTOFF.date()), "end": str(data_end.date())},
        "leakage_self_check": "passed — no recency-to-cutoff feature in feature set",
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {METRICS_OUT}")

    # ---- Final sanity gate ----
    if results[best_name]["auc"] >= 0.97:
        print("\nWARNING: best-model AUC >= 0.97 — this is suspiciously high for a "
              "behavioral churn model and likely indicates remaining leakage.")
    else:
        print(f"\nSanity gate OK: {best_name} AUC = {results[best_name]['auc']:.4f} "
              "is in a realistic behavioral-model range.")

    print("\nDone.")


if __name__ == "__main__":
    main()
