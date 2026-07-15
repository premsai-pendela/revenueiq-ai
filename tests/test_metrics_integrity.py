"""Regression guards on the model metrics — these fail loudly if a rebuild
reintroduces leakage, breaks the forecast baseline, or ships an ungrounded report."""
import json
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def _load(name):
    return json.loads((METRICS / name).read_text())


def test_churn_auc_is_honest_not_leaky():
    ch = _load("churn.json")
    # A leaky churn model (recency as feature + label) scores ~0.99-1.0.
    # An honest time-split model lands well below that.
    assert 0.5 <= ch["auc"] <= 0.95, f"AUC {ch['auc']} — ~1.0 signals leakage returned"
    assert ch["pr_auc"] > ch["churn_rate"], "PR-AUC should beat the base rate"


def test_segment_revenue_shares_sum_to_100():
    seg = _load("segmentation.json")
    total = sum(s["revenue_pct"] for s in seg["segments"])
    assert abs(total - 100.0) <= 1.0, f"segment revenue_pct sums to {total}"


def test_forecast_beats_seasonal_naive_baseline():
    fc = _load("forecasting.json")
    assert fc["mape"] < fc["baseline_mape"], "model must beat the seasonal-naive baseline"
    assert fc["improvement_vs_baseline_pct"] > 0


def test_report_is_fully_grounded():
    g = _load("report_grounding.json")
    assert g["ungrounded"] == 0, "the shipped AI report contains an ungrounded number"
    assert g["grounding_score"] == 1.0
