import fs from "fs";
import path from "path";
import { SegmentChart, MonthlyChart, ForecastChart, ShapChart, SegmentDonut, ClvScatter } from "@/components/Charts";
import { ReportView } from "@/components/ReportView";
import { Sidebar } from "@/components/Sidebar";
import { FindingCards } from "@/components/FindingCards";
import { usd, num, pct } from "@/lib/format";

import headline from "@/data/headline.json";
import segmentation from "@/data/segmentation.json";
import churn from "@/data/churn.json";
import forecasting from "@/data/forecasting.json";
import grounding from "@/data/report_grounding.json";
import monthly from "@/data/monthly_revenue.json";
import forecastSeries from "@/data/forecast_series.json";
import scatter from "@/data/clv_scatter.json";

function Spec({ items }: { items: [string, string][] }) {
  return (
    <div className="spec">
      {items.map(([k, v]) => (
        <div className="spec-i" key={k}><span>{k}</span><b>{v}</b></div>
      ))}
    </div>
  );
}

export default function Page() {
  const report = fs.readFileSync(path.join(process.cwd(), "data", "executive_report.md"), "utf8");
  const segs = [...segmentation.segments].sort((a, b) => b.revenue - a.revenue);
  const vip = segs[0];
  const tail = segs[segs.length - 1];
  const churnRate = churn.churn_rate * 100;
  const order = forecasting.chosen_order;
  const orderStr = `(${order.order.join(",")})(${order.seasonal_order.slice(0, 3).join(",")})[${order.seasonal_order[3]}]`;

  const findings = [
    { tag: "positive" as const, title: "Solid top line",
      body: `${usd(headline.total_revenue, true)} in net revenue across ${num(headline.total_customers)} customers and ${num(headline.total_orders)} orders.`,
      action: "Baseline is healthy — the risks below are where attention should go." },
    { tag: "risk" as const, title: "Revenue concentration",
      body: `${pct(vip.revenue_pct)} of revenue comes from just ${num(vip.count)} VIP customers (${pct(vip.customer_pct)} of the base).`,
      action: "Protect and widen the VIP tier — losing a few moves the top line sharply." },
    { tag: "risk" as const, title: "Customers slipping away",
      body: `${num(headline.high_risk_count)} customers are high-risk for churn, tied to ${usd(headline.high_risk_revenue, true)} of past revenue.`,
      action: "Run targeted win-back before the ~90-day inactivity cliff." },
    { tag: "opportunity" as const, title: "Under-monetized tail",
      body: `${tail.name} and low-frequency segments hold thousands of customers but a small share of revenue.`,
      action: "Activation nudges to lift second-purchase rate in the long tail." },
  ];

  return (
    <div className="app">
      <Sidebar />
      <div className="content">
        {/* ---------- HERO ---------- */}
        <header className="hero">
          <div className="hero-inner">
            <div className="kicker">RevenueIQ · Executive Intelligence</div>
            <h1>
              {usd(headline.total_revenue, true)} analyzed — but{" "}
              <span className="hl">{pct(vip.revenue_pct)}</span> of it rides on just{" "}
              <span className="hl">{pct(vip.customer_pct)}</span> of customers, while{" "}
              <span className="risk">{pct(churnRate)}</span> are churning.
            </h1>
            <p className="hero-sub">
              A governed retail-analytics pipeline: leakage-free churn prediction, behavioral
              segmentation, backtested forecasting, and an LLM executive report where every figure is
              verified against source data.
            </p>
          </div>
        </header>

        <main className="wrap">
          {/* ---------- Overview KPIs ---------- */}
          <section id="overview" className="block">
            <div className="kpis">
              <div className="kpi"><div className="v">{usd(headline.total_revenue, true)}</div><div className="l">Total net revenue</div><div className="sub">returns included</div></div>
              <div className="kpi"><div className="v">{num(headline.total_customers)}</div><div className="l">Customers</div><div className="sub">{num(headline.total_orders)} orders</div></div>
              <div className="kpi"><div className="v">{num(headline.high_risk_count)}</div><div className="l">High-risk customers</div><div className="sub">{usd(headline.high_risk_revenue, true)} at stake</div></div>
              <div className="kpi"><div className="v">{pct(vip.revenue_pct)}</div><div className="l">Revenue from top segment</div><div className="sub">{num(vip.count)} VIP customers</div></div>
            </div>
          </section>

          {/* ---------- Key Findings ---------- */}
          <section id="findings" className="block">
            <div className="section-head">
              <h2>Key findings</h2>
              <p>Lead with the health check, then the leaks — each with a number and an action.</p>
            </div>
            <FindingCards findings={findings} />
          </section>

          {/* ---------- Segments ---------- */}
          <section id="segments" className="block">
            <div className="section-head">
              <h2>Where the revenue actually comes from</h2>
              <p>
                KMeans on log-scaled RFM features. A thin VIP tier carries most of the revenue —
                the rest is concentration risk and untapped upside.
              </p>
            </div>
            <Spec items={[
              ["Model", `KMeans · k=${segmentation.chosen_k}`],
              ["Validation", `silhouette ${segmentation.silhouette.toFixed(2)} · ${pct(segmentation.stability_pct, 0)} stable`],
              ["Business use", "targeting · concentration risk"],
            ]} />
            <div className="grid-2">
              <div className="card"><SegmentChart segments={segs} /></div>
              <div className="card"><SegmentDonut segments={segs} /></div>
            </div>
            <div className="card" style={{ marginTop: 20 }}>
              <div className="l" style={{ color: "var(--muted)", fontSize: 13, marginBottom: 4 }}>Customer value landscape — value (log) vs. days since last purchase</div>
              <ClvScatter points={scatter as any} />
              <p className="note"><b>Read it:</b> high-value customers cluster at low recency (left). As recency grows (right), value drops — that drift toward the bottom-right is the churn leak in one picture.</p>
            </div>
          </section>

          {/* ---------- Churn ---------- */}
          <section id="churn" className="block">
            <div className="section-head">
              <h2>Who is about to leave — honestly measured</h2>
              <p>
                Churn model on a <b>time-split</b> design (features from the first ~10 months, outcome
                from the last ~3) so recency can’t leak the label. A real, defensible number.
              </p>
            </div>
            <Spec items={[
              ["Model", "XGBoost / RF / LogReg (compared)"],
              ["Validation", `PR-AUC ${churn.pr_auc.toFixed(2)} · leakage-checked`],
              ["Business use", "proactive retention"],
            ]} />
            <div className="grid-2">
              <div className="card">
                <div className="metric-row">
                  <div className="metric"><div className="v good">{churn.auc.toFixed(2)}</div><div className="l">ROC-AUC</div></div>
                  <div className="metric"><div className="v good">{churn.pr_auc.toFixed(2)}</div><div className="l">PR-AUC (base {churnRate.toFixed(0)}%)</div></div>
                  <div className="metric"><div className="v">{pct(churn.precision_at_200 * 100, 0)}</div><div className="l">Precision@200</div></div>
                  <div className="metric"><div className="v risk">{num(headline.high_risk_count)}</div><div className="l">flagged at-risk</div></div>
                </div>
                <p className="note" style={{ marginTop: 18 }}>
                  <b>Top churn-risk drivers (SHAP):</b> narrow product range, low order frequency, and
                  short tenure. Attribution, not proven causation.
                </p>
              </div>
              <div className="card"><ShapChart features={churn.top_shap_features} /></div>
            </div>
          </section>

          {/* ---------- Forecast ---------- */}
          <section id="forecast" className="block">
            <div className="section-head">
              <h2>Where revenue is heading</h2>
              <p>
                AutoARIMA validated by walk-forward backtest — it beats a seasonal-naive baseline,
                and the 30-day forecast is shown with its 95% band.
              </p>
            </div>
            <Spec items={[
              ["Model", `AutoARIMA ${orderStr}`],
              ["Validation", `MAPE ${pct(forecasting.mape)} · ${pct(forecasting.improvement_vs_baseline_pct)} better than naive`],
              ["Business use", "planning · staffing · inventory"],
            ]} />
            <div className="grid-2 even">
              <div className="card">
                <div className="metric-row" style={{ marginBottom: 10 }}>
                  <div className="metric"><div className="v">{usd(forecasting.next30_total, true)}</div><div className="l">next 30 days (forecast)</div></div>
                  <div className="metric"><div className="v" style={{ fontSize: 15, color: "var(--muted)" }}>{usd(forecasting.interval_low, true)} – {usd(forecasting.interval_high, true)}</div><div className="l">95% range</div></div>
                </div>
                <ForecastChart series={forecastSeries} />
              </div>
              <div className="card">
                <div className="l" style={{ color: "var(--muted)", fontSize: 13, marginBottom: 6 }}>Monthly revenue (13 months)</div>
                <MonthlyChart data={monthly} />
                <p className="note">Only ~13 months — weekly seasonality is learned; a full yearly cycle can be described but not validated.</p>
              </div>
            </div>
          </section>

          {/* ---------- Grounded LLM report ---------- */}
          <section id="report" className="block">
            <div className="section-head guard-head">
              <div>
                <h2>The AI executive report — every number verified</h2>
                <p>
                  An LLM writes this brief, but it may only use numbers computed above. A grounding guard
                  checks every figure against source data and regenerates any it can’t trace.
                </p>
              </div>
              <span className={grounding.ungrounded === 0 ? "pill ok" : "pill warn"}>
                <span className="dot" />
                {grounding.grounded}/{grounding.checked} figures verified · score {grounding.grounding_score}
              </span>
            </div>
            <ReportView markdown={report} />
          </section>

          <footer>
            <div className="stack">Python · DuckDB · scikit-learn · XGBoost · SHAP · statsforecast · Groq LLM · Next.js · Recharts</div>
            <div>
              Data: UCI Online Retail ({num(headline.total_lines)} transactions, Dec 2010 – Dec 2011). A single public
              dataset — a demonstration of method, not a live production system. Every figure on this page is
              reproducible from the model metrics in the repo.
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
