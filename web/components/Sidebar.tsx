"use client";

import { useEffect, useState } from "react";

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "findings", label: "Key Findings" },
  { id: "segments", label: "Segments" },
  { id: "churn", label: "Churn" },
  { id: "forecast", label: "Forecast" },
  { id: "report", label: "AI Report" },
];

const HIGHLIGHTS = [
  "534,117 transactions modelled",
  "3 leakage-checked ML models",
  "7.7× faster SQL (DuckDB vs Pandas)",
  "LLM report, every figure verified",
];

const STACK = [
  ["Data", "Python · DuckDB · pandas"],
  ["ML", "scikit-learn · XGBoost · SHAP"],
  ["Forecast", "statsforecast (AutoARIMA)"],
  ["AI", "Groq LLM + grounding guard"],
  ["Web", "Next.js · Recharts"],
];

export function Sidebar() {
  const [active, setActive] = useState("overview");

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => e.isIntersecting && setActive(e.target.id));
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    NAV.forEach((n) => {
      const el = document.getElementById(n.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  return (
    <aside className="sidebar">
      <div className="sb-brand">
        <span className="sb-logo">◑</span>
        <div>
          <div className="sb-name">RevenueIQ</div>
          <div className="sb-tag">Executive Intelligence</div>
        </div>
      </div>

      <nav className="sb-nav">
        {NAV.map((n) => (
          <a key={n.id} href={`#${n.id}`} className={active === n.id ? "active" : ""}>
            <span className="sb-dot" />
            {n.label}
          </a>
        ))}
      </nav>

      <div className="sb-section">
        <div className="sb-h">Project highlights</div>
        <ul className="sb-list">
          {HIGHLIGHTS.map((h) => <li key={h}>{h}</li>)}
        </ul>
      </div>

      <div className="sb-section">
        <div className="sb-h">Tech stack</div>
        {STACK.map(([k, v]) => (
          <div className="sb-stack" key={k}>
            <span className="sk">{k}</span>
            <span className="sv">{v}</span>
          </div>
        ))}
      </div>

      <div className="sb-section sb-contact">
        <div className="sb-h">Naga Prem Sai Pendela</div>
        <a href="https://github.com/premsai-pendela/revenueiq-ai" target="_blank" rel="noreferrer">↗ GitHub repo</a>
        <a href="https://linkedin.com/in/nagapremsai-pendela" target="_blank" rel="noreferrer">↗ LinkedIn</a>
        <a href="mailto:nagapremsaip07@gmail.com">✉ nagapremsaip07@gmail.com</a>
      </div>
    </aside>
  );
}
