"""
Export everything the Next.js dashboard needs into web/data/*.json.
All numbers are REAL — pulled from the rebuilt model metrics + DuckDB.
Run: ./venv/bin/python scripts/export_dashboard_data.py
"""
import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "outputs" / "metrics"
WEB_DATA = REPO / "web" / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)
DB = REPO / "data" / "revenueiq.db"


def jdump(name, obj):
    with open(WEB_DATA / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  ✓ {name}")


def main():
    print("Exporting dashboard data → web/data/")

    # 1) Copy the model metric files + grounding audit verbatim
    for f in ["churn.json", "segmentation.json", "forecasting.json", "report_grounding.json"]:
        shutil.copy(METRICS / f, WEB_DATA / f)
        print(f"  ✓ {f} (copied)")

    # 2) Copy the grounded executive report
    shutil.copy(REPO / "outputs" / "reports" / "executive_report.md", WEB_DATA / "executive_report.md")
    print("  ✓ executive_report.md (copied)")

    con = duckdb.connect(str(DB), read_only=True)

    # 3) Headline KPIs
    row = con.execute(
        "SELECT COUNT(*) n, COUNT(DISTINCT CustomerID) c, COUNT(DISTINCT InvoiceNo) o, "
        "SUM(TotalPrice) rev, AVG(TotalPrice) aov FROM transactions"
    ).fetchone()
    ar = pd.read_csv(REPO / "data" / "processed" / "at_risk_customers.csv")
    hi = ar[ar["churn_prob"] > 0.7]
    jdump("headline.json", {
        "total_revenue": round(row[3], 2),
        "total_customers": int(row[1]),
        "total_orders": int(row[2]),
        "total_lines": int(row[0]),
        "avg_order_line": round(row[4], 2),
        "high_risk_count": int(len(hi)),
        "high_risk_revenue": round(float(hi["revenue_at_risk"].sum()), 2),
    })

    # 3b) CLV vs Recency scatter (sampled, stratified by segment)
    clus = pd.read_csv(REPO / "data" / "processed" / "customer_clusters.csv")
    clus = clus[(clus["MonetaryValue"] > 0)]
    sample = (clus.groupby("ClusterName", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 180), random_state=42)))
    scatter = [{
        "recency": int(r.Recency),
        "monetary": round(float(r.MonetaryValue), 2),
        "frequency": int(r.Frequency),
        "segment": r.ClusterName,
    } for r in sample.itertuples()]
    jdump("clv_scatter.json", scatter)

    # 4) Monthly revenue history (for the interactive bar chart)
    mrev = con.execute(
        "SELECT YearMonth AS month, ROUND(SUM(TotalPrice),2) AS revenue "
        "FROM transactions GROUP BY YearMonth ORDER BY YearMonth"
    ).df()
    jdump("monthly_revenue.json", mrev.to_dict(orient="records"))

    # 5) Daily series + 30-day forecast with 95% band (for the forecast chart)
    daily = con.execute(
        "SELECT CAST(InvoiceDate AS DATE) AS d, SUM(TotalPrice) AS revenue "
        "FROM transactions GROUP BY d ORDER BY d"
    ).df()
    con.close()
    daily["d"] = pd.to_datetime(daily["d"])
    daily = daily.set_index("d").asfreq("D").fillna(0.0)

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        sdf = daily.reset_index().rename(columns={"d": "ds", "revenue": "y"})
        sdf["unique_id"] = "revenue"
        sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D")
        sf.fit(sdf[["unique_id", "ds", "y"]])
        fc = sf.predict(h=30, level=[95]).reset_index()
        lo_col = [c for c in fc.columns if "lo-95" in c][0]
        hi_col = [c for c in fc.columns if "hi-95" in c][0]
        mean_col = "AutoARIMA"
        forecast = [{
            "date": str(pd.to_datetime(r["ds"]).date()),
            "forecast": max(0.0, round(float(r[mean_col]), 2)),
            "lo": max(0.0, round(float(r[lo_col]), 2)),
            "hi": round(float(r[hi_col]), 2),
        } for _, r in fc.iterrows()]
        note = "AutoARIMA(season_length=7), 95% interval"
    except Exception as e:  # honest fallback
        forecast = []
        note = f"forecast series unavailable ({type(e).__name__})"

    hist = [{"date": str(d.date()), "actual": round(float(v), 2)}
            for d, v in daily["revenue"].tail(90).items()]
    jdump("forecast_series.json", {"note": note, "history": hist, "forecast": forecast})

    print("\nDone. Dashboard data is in web/data/.")


if __name__ == "__main__":
    main()
