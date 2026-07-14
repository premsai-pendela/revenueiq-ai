"""
RevenueIQ AI — Grounded Executive Report (LLM + verification guard)
===================================================================

WHY THIS EXISTS
---------------
The old pipeline (`src/groq_insights.py`) hard-coded stale numbers into the
prompt and never checked the LLM's output. On the deployed site the "AI
insights" were literally hard-coded strings. So the LLM was free to invent
dollar figures in executive-facing text — exactly the failure a reviewer
flagged ("how do you stop the LLM hallucinating numbers?").

WHAT THIS DOES
--------------
1. Loads ONLY real, reproducible numbers from `outputs/metrics/*.json`
   (produced by the rebuilt ML models) plus live DuckDB aggregates.
   This is the FACT REGISTRY — the sole set of numbers the LLM may use.
2. Asks Groq to write an executive report using ONLY those facts, structured
   the way a real analyst pitches: a short health check, then the leaks/risks,
   then actions.
3. GROUNDING GUARD: extracts every monetary value, percentage, and count>=100
   from the LLM output and verifies each one traces back to a fact (within a
   rounding tolerance) or is explicitly labelled an assumption. Anything
   ungrounded is a hallucination.
4. If hallucinations are found, it sends them back once and regenerates, then
   re-verifies. The final report + a full grounding audit are saved.

This is a BATCH, REPRODUCIBLE pipeline — not an "autonomous/real-time" agent.
Run: ./venv/bin/python src/llm/grounded_report.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs" / "metrics"
DB = REPO / "data" / "revenueiq.db"
AT_RISK_CSV = REPO / "data" / "processed" / "at_risk_customers.csv"
REPORT_OUT = REPO / "outputs" / "reports" / "executive_report.md"
AUDIT_OUT = METRICS / "report_grounding.json"

MODEL = "llama-3.3-70b-versatile"

# Matching tolerances for the grounding guard
CURRENCY_REL_TOL = 0.03   # 3% (covers "$5.5M" vs $5,537,125)
PERCENT_ABS_TOL = 1.0     # 1 percentage point (covers rounding 66.5% -> 67%)
COUNT_REL_TOL = 0.02      # 2% (covers "~3,600" vs 3,604)
RATIO_ABS_TOL = 0.02      # AUC-style 0..1 values


# --------------------------------------------------------------------------- #
# FACT REGISTRY                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    fact_id: str
    label: str
    value: float
    kind: str  # currency | percent | count | ratio


def _load_json(name: str) -> dict:
    with open(METRICS / name) as f:
        return json.load(f)


def build_fact_registry() -> list[Fact]:
    facts: list[Fact] = []

    # --- Live business aggregates from DuckDB (the canonical store) ---
    con = duckdb.connect(str(DB), read_only=True)
    agg = con.execute(
        "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT CustomerID) AS custs, "
        "COUNT(DISTINCT InvoiceNo) AS orders, ROUND(SUM(TotalPrice),2) AS rev, "
        "ROUND(AVG(TotalPrice),2) AS aov FROM transactions"
    ).fetchone()
    con.close()
    n_rows, custs, orders, rev, aov = agg
    facts += [
        Fact("total_revenue", "Total net revenue (returns included)", float(rev), "currency"),
        Fact("total_customers", "Total customers", float(custs), "count"),
        Fact("total_orders", "Total orders", float(orders), "count"),
        Fact("total_transactions", "Total transaction lines", float(n_rows), "count"),
        Fact("avg_order_line", "Average transaction-line value", float(aov), "currency"),
    ]

    # --- Segmentation ---
    seg = _load_json("segmentation.json")
    facts.append(Fact("seg_k", "Number of customer segments", float(seg["chosen_k"]), "count"))
    facts.append(Fact("seg_silhouette", "Segmentation silhouette score", float(seg["silhouette"]), "ratio"))
    for s in seg["segments"]:
        sid = re.sub(r"[^a-z0-9]+", "_", s["name"].lower()).strip("_")
        facts.append(Fact(f"seg_{sid}_count", f"{s['name']} — customer count", float(s["count"]), "count"))
        facts.append(Fact(f"seg_{sid}_rev", f"{s['name']} — revenue", float(s["revenue"]), "currency"))
        facts.append(Fact(f"seg_{sid}_revpct", f"{s['name']} — % of revenue", float(s["revenue_pct"]), "percent"))

    # --- Churn ---
    ch = _load_json("churn.json")
    facts += [
        Fact("churn_auc", "Churn model ROC-AUC", float(ch["auc"]), "ratio"),
        Fact("churn_pr_auc", "Churn model PR-AUC", float(ch["pr_auc"]), "ratio"),
        Fact("churn_rate", "Churn rate (outcome window)", float(ch["churn_rate"]) * 100, "percent"),
        Fact("churn_customers", "Customers modelled for churn", float(ch["n_customers"]), "count"),
        Fact("churn_prec200", "Precision@top-200 at-risk", float(ch["precision_at_200"]) * 100, "percent"),
        Fact("churn_prec_k", "Precision measured at the top-K riskiest customers (K)", 200.0, "count"),
    ]

    # High-risk subset (prob > 0.7) revenue-at-risk — computed honestly here
    import pandas as pd
    ar = pd.read_csv(AT_RISK_CSV)
    hi = ar[ar["churn_prob"] > 0.7]
    facts += [
        Fact("high_risk_count", "High-risk customers (churn prob > 0.7)", float(len(hi)), "count"),
        Fact("high_risk_revenue", "Historical revenue tied to high-risk customers",
             float(round(hi["revenue_at_risk"].sum(), 2)), "currency"),
    ]

    # --- Forecasting ---
    fc = _load_json("forecasting.json")
    facts += [
        Fact("fc_mape", "Forecast MAPE (walk-forward)", float(fc["mape"]), "percent"),
        Fact("fc_baseline_mape", "Seasonal-naive baseline MAPE", float(fc["baseline_mape"]), "percent"),
        Fact("fc_improvement", "Forecast improvement vs baseline", float(fc["improvement_vs_baseline_pct"]), "percent"),
        Fact("fc_next30", "Forecast: next 30 days total revenue", float(fc["next30_total"]), "currency"),
        Fact("fc_low", "Forecast 95% interval low", float(fc["interval_low"]), "currency"),
        Fact("fc_high", "Forecast 95% interval high", float(fc["interval_high"]), "currency"),
    ]
    return facts


# --------------------------------------------------------------------------- #
# NUMBER EXTRACTION + NORMALISATION                                           #
# --------------------------------------------------------------------------- #
SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}

CURRENCY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?([kmbKMB])?")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")
NUMBER_RE = re.compile(r"(?<![\w$.])(\d[\d,]*(?:\.\d+)?)(?![\w%])")

# Structural numbers we do NOT treat as business claims
IGNORE_EXACT = {2009, 2010, 2011, 2012, 2013}


@dataclass
class Extracted:
    raw: str
    value: float
    kind: str


def _to_float(num_str: str, suffix: str | None = None) -> float:
    v = float(num_str.replace(",", ""))
    if suffix:
        v *= SUFFIX[suffix.lower()]
    return v


def extract_numbers(text: str) -> list[Extracted]:
    """Extract currency, percent, and large counts. Mask matches as we go so a
    single token isn't double-counted."""
    found: list[Extracted] = []
    masked = text

    for m in CURRENCY_RE.finditer(text):
        found.append(Extracted(m.group(0), _to_float(m.group(1), m.group(2)), "currency"))
    masked = CURRENCY_RE.sub(" ", masked)

    for m in PERCENT_RE.finditer(masked):
        found.append(Extracted(m.group(0), float(m.group(1)), "percent"))
    masked = PERCENT_RE.sub(" ", masked)

    for m in NUMBER_RE.finditer(masked):
        v = _to_float(m.group(1))
        if "." in m.group(1) and v < 1.5:
            found.append(Extracted(m.group(0), v, "ratio"))       # AUC-style
        elif v >= 100 and int(v) not in IGNORE_EXACT:
            found.append(Extracted(m.group(0), v, "count"))       # business counts
        # small integers / years are structural -> ignored
    return found


# --------------------------------------------------------------------------- #
# GROUNDING GUARD                                                             #
# --------------------------------------------------------------------------- #
def _within_tol(ex: Extracted, f: Fact) -> bool:
    if ex.kind == "currency":
        return abs(ex.value - f.value) / max(abs(f.value), 1.0) <= CURRENCY_REL_TOL
    if ex.kind == "percent":
        return abs(ex.value - f.value) <= PERCENT_ABS_TOL
    if ex.kind == "count":
        return abs(ex.value - f.value) / max(abs(f.value), 1.0) <= COUNT_REL_TOL
    if ex.kind == "ratio":
        return abs(ex.value - f.value) <= RATIO_ABS_TOL
    return False


def _match(ex: Extracted, facts: list[Fact]) -> Fact | None:
    """Return the CLOSEST fact of the same kind within tolerance (not first-match),
    so numerically-adjacent facts (e.g. 700 vs 707) attribute to the right source."""
    candidates = [f for f in facts if f.kind == ex.kind and _within_tol(ex, f)]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(ex.value - f.value))


def _is_assumption(raw: str, text: str) -> bool:
    """True if the number is explicitly framed as an assumption/estimate."""
    idx = text.find(raw)
    if idx == -1:
        return False
    window = text[max(0, idx - 60): idx + 60].lower()
    return any(w in window for w in ("assume", "assumption", "estimate", "hypothetical",
                                     "for illustration", "e.g.", "example", "if we"))


def verify(text: str, facts: list[Fact]) -> dict:
    results = []
    for ex in extract_numbers(text):
        hit = _match(ex, facts)
        if hit:
            verdict = "grounded"
            src = hit.fact_id
        elif _is_assumption(ex.raw, text):
            verdict = "assumption"
            src = None
        else:
            verdict = "UNGROUNDED"
            src = None
        results.append({"raw": ex.raw, "value": ex.value, "kind": ex.kind,
                        "verdict": verdict, "source_fact": src})
    ungrounded = [r for r in results if r["verdict"] == "UNGROUNDED"]
    checked = len(results)
    grounded = sum(1 for r in results if r["verdict"] == "grounded")
    return {
        "checked": checked,
        "grounded": grounded,
        "assumptions": sum(1 for r in results if r["verdict"] == "assumption"),
        "ungrounded": len(ungrounded),
        "grounding_score": round(grounded / checked, 3) if checked else 1.0,
        "scope": "verifies all currency, percentage, and count>=100 figures; "
                 "years and small integers treated as structural",
        "numbers": results,
    }


# --------------------------------------------------------------------------- #
# LLM                                                                         #
# --------------------------------------------------------------------------- #
def facts_block(facts: list[Fact]) -> str:
    lines = []
    for f in facts:
        if f.kind == "currency":
            x = f.value
            if x >= 1_000_000:
                v = f"${x/1e6:.2f}M"
            elif x >= 1_000:
                v = f"${x/1e3:.0f}K"
            else:
                v = f"${x:,.2f}"
        elif f.kind == "percent":
            v = f"{f.value:.1f}%"
        elif f.kind == "ratio":
            v = f"{f.value:.3f}"
        else:
            v = f"{int(round(f.value)):,}"
        lines.append(f"- [{f.fact_id}] {f.label}: {v}")
    return "\n".join(lines)


SYSTEM = (
    "You are a senior data analyst pitching findings to a business owner. "
    "You write plainly and honestly. CRITICAL RULE: you may only cite numbers "
    "that appear in the FACTS list. Do NOT invent, extrapolate, or compute new "
    "dollar figures, percentages, or counts. If you want to state a projected "
    "impact that is not in FACTS, you must explicitly label it 'assumption'. "
    "Do not restate a fact with a different number."
)


def user_prompt(facts: list[Fact], violations: list[str] | None = None) -> str:
    base = f"""Write a one-page executive report for the business owner using ONLY the FACTS below.

FACTS (the only numbers you may use):
{facts_block(facts)}

Format as GitHub-flavored markdown. Start directly with the first heading (no title line).
Use these exact section headings:

## Health check
2-3 sentences: the positives — revenue, customers, orders.

## Where you are losing
The bulk of the report: churn and the revenue at risk, the thin VIP revenue concentration,
the dormant / low-value segments, and the forecast outlook. Frame each as a problem with its
number attached. Use short paragraphs.

## Recommended actions
A markdown bullet list ('- ') of 3-4 specific actions tied to the findings above.

Rules:
- Every number must come from FACTS. No new numbers. Use the compact form exactly as shown (e.g. $9.87M, $320K).
- Plain language, no hype, no invented dollar impacts.
- If you must estimate an impact, prefix it with 'assumption:'."""
    if violations:
        base += (
            "\n\nYOUR PREVIOUS DRAFT CONTAINED NUMBERS NOT IN FACTS: "
            + ", ".join(violations)
            + ".\nRewrite so every number is either from FACTS or explicitly labelled 'assumption'."
        )
    return base


def call_llm(client: Groq, facts: list[Fact], violations=None) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user_prompt(facts, violations)}],
        temperature=0.3,
        max_tokens=1600,
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("REVENUEIQ — GROUNDED EXECUTIVE REPORT (with verification guard)")
    print("=" * 70)

    facts = build_fact_registry()
    print(f"\n✓ Fact registry built: {len(facts)} verified source numbers")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Generate, verify, and regenerate against violations — up to MAX_PASSES.
    MAX_PASSES = 3
    violations = None
    best_report, best_audit = None, None
    passes_used = 0
    for p in range(1, MAX_PASSES + 1):
        passes_used = p
        report = call_llm(client, facts, violations=violations)
        audit = verify(report, facts)
        print(f"Pass {p} grounding: {audit['grounded']}/{audit['checked']} grounded, "
              f"{audit['ungrounded']} ungrounded (score {audit['grounding_score']})")
        if best_audit is None or audit["ungrounded"] < best_audit["ungrounded"]:
            best_report, best_audit = report, audit
        if audit["ungrounded"] == 0:
            break
        violations = [r["raw"] for r in audit["numbers"] if r["verdict"] == "UNGROUNDED"]
        print(f"  ↳ ungrounded: {violations} — regenerating...")

    report, audit = best_report, best_audit
    audit["passes"] = passes_used
    audit["model"] = MODEL

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ("# RevenueIQ — Executive Report\n"
              f"_Generated by {MODEL}; every figure verified against source metrics "
              f"(grounding score {audit['grounding_score']}, {audit['ungrounded']} ungrounded)._\n\n")
    REPORT_OUT.write_text(header + report)
    with open(AUDIT_OUT, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n✓ Report saved:  {REPORT_OUT}")
    print(f"✓ Audit saved:   {AUDIT_OUT}")
    if audit["ungrounded"] == 0:
        print(f"\n✅ VERIFIED: all {audit['checked']} figures trace to source metrics.")
    else:
        print(f"\n⚠️  {audit['ungrounded']} figure(s) still ungrounded — flagged in audit, "
              "NOT silently shipped.")


if __name__ == "__main__":
    main()
