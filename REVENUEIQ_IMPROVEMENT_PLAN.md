# RevenueIQ AI — Improvement Plan (Agentic Rebuild)

> Owner: Prem. Author of plan: senior-ML review + Murta's resume review, combined.
> Purpose: turn RevenueIQ from a portfolio demo with hidden flaws into an **honest, defensible, keyword-rich** applied-ML project.
> Golden rule: **every number on the resume must be reproducible from this code.** No fake metrics, no fake "live AI," no padding.

---

## 0. Why we are doing this (the negatives we found)

Deep review of the actual code (not the docs) found:

1. **Churn model has target leakage.** `Churned` is *defined* as `Recency > 90`, and `Recency` is *also* a feature. The model predicts the answer it was handed. The "95% F1" is fake — a one-line rule scores the same. `Recency` = 81% importance confirms it. **This is the #1 fix.**
2. **"5 ML models" is padded.** Product "demand forecast" is a 7-day moving average (not a model). Anomaly detection is circular (`contamination=0.01` → "we found 1%"). Only churn, segmentation, forecasting are substantive.
3. **KMeans weaknesses.** No log-transform on skewed money/frequency features (whales distort clusters); `find_optimal_clusters()` computes silhouette then hard-returns `5` regardless (fake "data-driven k"); fragile threshold-based cluster naming.
4. **Forecasting weaknesses.** ARIMA order hard-coded `(1,1,1)`; single 80/20 split; no baseline comparison; no prediction intervals. Docs claim "Prophet" but code uses ARIMA + ExponentialSmoothing (claim ≠ code).
5. **LLM is fake-live on the deployed site.** `app.py`'s "AI-Generated Insights" tab shows **hard-coded text**; `groq` and `duckdb` are **not in `requirements.txt`**. The LLM only ran once offline to make `.txt` files; the dashboard displays static copy labeled "auto-generated." **This is the biggest honesty problem because it's on the live site.**
6. **No grounding guard on LLM output.** Even offline, the LLM invents projected dollars ("$96K–144K recoverable") at temperature 0.5–0.7 with no check that output numbers trace to source numbers. This is exactly what Murta flagged ("how do you stop it hallucinating numbers in executive output").

## Decisions locked

- **KEEP + deepen:** Churn, Segmentation (KMeans), Forecasting.
- **CUT entirely:** Product demand forecast.
- **DROP from resume:** Anomaly detection (may stay in repo as a minor "outlier review," but earns no resume line).
- **DB:** keep **DuckDB** (correct for 70MB, trendy) + add **dbt** (dbt-duckdb) for analytics-engineering keyword and real data-quality tests. **No Spark** (red flag for 70MB). BigQuery/Snowflake only if genuinely run.
- **Dashboard:** drop Streamlit + Plotly → lean **Next.js** report page (consistent with NexusIQ; renders the *real* grounded LLM report). No Power BI (Windows-only, analyst-cert-coded, wrong target).
- **LLM:** keep "LLM writes report from computed numbers" direction; add a **grounding/verification guard**; render the real artifact (not hard-coded). **Batch/reproducible run — NOT "autonomous/real-time"** (don't claim autonomy we don't have).

## Target role

AI/ML engineer roles (not data-analyst — those want Power BI + certs + 2–3 yrs). This project is the "data→AI breadth" project; NexusIQ is the "AI systems" project.

---

## 1. Shared foundation (parent-owned; do NOT let model agents edit these)

- **Data source:** `data/processed/transactions_sales_only.csv` (524,875 rows, returns excluded) and `transactions_cleaned.csv` (534K, returns flagged).
- **Real columns:** `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, IsReturn, TotalPrice, Year, Month, Day, DayOfWeek, Hour, MonthName, DayName, YearMonth`. Date range Dec 2010 – Dec 2011 (~13 months). `CustomerID` may be `'Guest'` or float; exclude Guests for customer-level models.
- **Env:** `./venv/bin/python` (Python 3.13, pandas 3.0, sklearn 1.8, statsmodels, duckdb). Parent installs: `xgboost, shap, statsforecast, groq`.
- **New module layout (each agent owns exactly ONE new file):**
  - `src/models/churn_model.py`
  - `src/models/segmentation_model.py`
  - `src/models/forecasting_model.py`
- **Output contract (every model writes a small JSON of its real metrics):** `outputs/metrics/<model>.json` so the resume/LLM can read *actual* numbers. Never hard-code metrics anywhere else.
- **Honesty contract:** report the real number you get, even if "worse" than the old fake. A real 0.82 AUC beats a fake 0.95.

---

## 2. MODEL 1 — Churn Prediction (owner: Agent A)

**Business job:** flag customers about to stop buying, with the revenue at risk. This is the core "where are you losing money" model.

**Current (broken):** `predictive_models.py` `predict_customer_churn()` — leakage.

**Rebuild spec → `src/models/churn_model.py`:**
1. **Time-split design (removes leakage):**
   - Observation window = first ~10 months; Outcome window = last ~3 months (pick a cutoff date, e.g. `2011-09-30`).
   - Build features using **only** the observation window.
   - Label `churned = 1` if the customer made **no purchase in the outcome window** (given they were active in observation). Exclude customers not present in observation.
   - **Do NOT use recency-to-dataset-end as a feature.** Recency may be measured relative to the observation-window end only.
2. **Features (observation window only):** frequency (# orders), monetary (sum), avg order value, avg inter-purchase gap (days), std of gap, tenure, distinct products, total quantity, spend trend (last 30d vs prior 30d within observation).
3. **Models:** Logistic Regression baseline → RandomForest → **XGBoost**. Compare all three.
4. **Evaluation (honest):** ROC-AUC, **PR-AUC** (primary, imbalanced), precision@top-N (e.g., top 200 riskiest), confusion matrix. Stratified split or time-consistent validation. Report class balance.
5. **Explainability:** **SHAP** — global feature importance + a short "top drivers of churn risk" list. State clearly it is correlation/attribution, not proven causation.
6. **Outputs:** `outputs/metrics/churn.json` (auc, pr_auc, precision_at_200, churn_rate, n_customers, best_model, top_shap_features), `data/processed/at_risk_customers.csv` (customer, prob, revenue_at_risk), a SHAP summary plot PNG.
7. **Verify:** run end-to-end on real data; AUC should land in a realistic ~0.75–0.90 band (NOT ~1.0 — 1.0 means leakage remains). Print a leakage self-check (confirm recency-to-end not in feature list).

**Definition of done:** honest metrics in JSON, no leakage, XGBoost + SHAP working, at-risk list with revenue-at-risk.

---

## 3. MODEL 2 — Customer Segmentation (owner: Agent B)

**Business job:** group customers by behavior → show the thin VIP layer carrying revenue and the big under-monetized segment.

**Current:** `kmeans_clustering.py` — works but no log-transform, fake k-selection, fragile naming.

**Rebuild spec → `src/models/segmentation_model.py`:**
1. **Features:** RFM core + AOV + tenure + quantity (per customer, full period, exclude Guests, monetary > 0).
2. **Fix skew:** apply `log1p` to Monetary, Frequency, Quantity, AOV **before** StandardScaler. (This is the #1 fix — whales must not dominate.)
3. **Honest k selection:** compute inertia + silhouette for k=2..10; pick k by elbow+silhouette; if you keep k=5, justify it explicitly in output — do NOT hard-return 5 while pretending it's data-driven.
4. **Judgment:** short comparison note KMeans vs classic RFM-quantile scoring (why KMeans surfaces the ultra-VIP tier). Optional: also fit GaussianMixture and report which is cleaner.
5. **Robust naming:** name clusters by ranking their real stats (revenue, recency, frequency), not fragile fixed thresholds; guarantee unique names.
6. **Cluster stability:** re-fit with 2–3 seeds, report label agreement (stability %).
7. **Outputs:** `outputs/metrics/segmentation.json` (chosen_k, silhouette, stability, per-segment: count, revenue, revenue_pct, avg_recency/frequency/monetary, name), `data/processed/customer_clusters.csv`, DuckDB table `customer_clusters`, 2D/3D plots.
8. **Verify:** run on real data; confirm segments are sensible (a small high-value segment + a large low-frequency segment), silhouette printed, stability printed.

**Definition of done:** log-transformed, honest k, stable, named-from-data, JSON metrics.

---

## 4. MODEL 3 — Sales Forecasting (owner: Agent C)

**Business job:** project near-term revenue + seasonal swing so the owner plans for the dip. Supporting model.

**Current:** ARIMA(1,1,1) hard-coded + Holt-Winters; single split; no baseline; docs falsely say Prophet.

**Rebuild spec → `src/models/forecasting_model.py`:**
1. **Series:** daily total revenue from transactions (fill missing days). Note only ~13 months → yearly seasonality can be *described* not *learned*; say so honestly.
2. **Stationarity:** ADF test; difference as needed.
3. **Auto model selection:** prefer **statsforecast `AutoARIMA`** (modern, fast, trending). Fallback: statsmodels with a small order grid, or Holt-Winters. Do NOT hard-code `(1,1,1)`.
4. **Baselines (critical):** naive (last value) + **seasonal-naive** (same weekday last week). Your model must BEAT these; report by how much.
5. **Backtesting:** walk-forward / rolling-origin validation (multiple folds), not one 80/20. Report MAE, RMSE, MAPE averaged over folds.
6. **Prediction intervals:** forecast next 30 days **with** confidence bands, not a bare point.
7. **Outputs:** `outputs/metrics/forecasting.json` (chosen_order/model, mae, rmse, mape, baseline_mape, improvement_vs_baseline_pct, next30_total, interval_low, interval_high), forecast plot with bands.
8. **Verify:** run on real data; confirm model beats seasonal-naive (if it doesn't, say so honestly — that's a real finding).

**Definition of done:** auto-selected model, beats baseline (or honest note), walk-forward metrics, intervals, JSON.

---

## 5. LLM grounding guard (PHASE 2) — ✅ DONE 2026-07-14

- Built `src/llm/grounded_report.py`: loads a **fact registry** (35 real numbers from `outputs/metrics/*.json` + live DuckDB aggregates); Groq writes the exec report using ONLY those facts; a **grounding guard** extracts every currency/percent/count≥100 figure and verifies it against the registry (closest-match, rounding tolerances) or an explicit "assumption"; ungrounded numbers are fed back and regenerated (up to 3 passes).
- **Demonstrated live catch:** Pass 1 the LLM invented a "95%"; guard flagged it; Pass 2 → **17/17 grounded, score 1.0**. Full audit at `outputs/metrics/report_grounding.json`; report at `outputs/reports/executive_report.md`.
- Report structure follows the analyst pitch: short health check → the leaks/risks (churn, at-risk revenue, VIP concentration, dormant segments, forecast) → actions.
- `requirements.txt` fixed: added `duckdb, xgboost, shap, statsforecast, groq` (were missing — why the deployed app couldn't run the LLM/DB layer).
- Batch + reproducible (`./venv/bin/python src/llm/grounded_report.py`). **Not** claimed as autonomous/real-time.

## 6. dbt + DuckDB (PHASE 3)

- Add `dbt-duckdb`; model the SQL transformations as dbt models with `tests` (not null, unique, accepted ranges). Real analytics-engineering + data-quality keyword.

## 7. Next.js dashboard (PHASE 4) — ✅ DONE 2026-07-14

- Built `web/` — Next.js 16 + React 19 + Recharts. **Two-tone calm theme**: dark hero band with the headline finding, light cool body, single teal accent (+ muted rose for loss/at-risk figures). Distinct from the warm-cream NexusIQ/portfolio look.
- Interactive charts (hover tooltips): segment revenue bars, monthly revenue, 90-day history + 30-day forecast with 95% band, SHAP churn drivers.
- Renders the **grounded LLM report** live with a "17/17 figures verified · score 1.0" badge (the real audit).
- Data via `scripts/export_dashboard_data.py` → `web/data/*.json` (all real, reproducible).
- `npm run build` passes; `/` **prerenders static** → free Vercel deploy, no server, no secrets.
- Deploy: push repo → Vercel → New Project → **Root Directory = `web`** → Next.js auto-detected → Deploy. No env vars.
- **Architecture pass (adopted from the old Streamlit demo's structure, honest data):** fixed dark **sidebar** (brand, scroll-spy nav, honest project highlights, tech stack, real GitHub/LinkedIn/email links); **Key Findings** cards (finding → number → action); per-model **spec strips** (Model / Validation / Business use); added **segment donut** + **CLV-vs-recency scatter** charts. Deliberately did NOT port the old demo's dishonest bits (fake YoY arrows, 95% leaky-churn, invented $ impacts, broken LinkedIn link).

## 8. Then: resume rewrite (PHASE 5)

Using the REAL metrics now in `outputs/metrics/*.json`, rewrite RevenueIQ bullets to satisfy Murta:
- **Action verb + what + process/how + real metric** (#10), each **≤ 2 lines** (#4), **plain non-GPT voice** (#1), no "analyzed revenue"/activity lists (#6), grounding guard answers the hallucination worry (#8).
- Separately fix the NLP research bullet (#9) — outside RevenueIQ.

---

## Parallelization & safety (this run)

- **Parallel:** Agents A/B/C run at once — disjoint files, disjoint outputs. Each reads this plan + its old code first, then implements, then verifies on real data.
- **Do NOT** let agents edit `requirements.txt`, each other's files, or run the full shared pipeline. Deps pre-installed by parent.
- Each agent returns: file created, real metrics JSON, verification output, and any honest caveats.
- Parent verifies each: reruns/inspects metrics, checks for leakage / sane numbers, then integrates. Phases 2–5 after models are green.

## Verification checklist (parent, after agents return)

- [x] Churn: **ROC-AUC 0.72 / PR-AUC 0.68** (leakage-free; real ceiling, not ~1.0), leakage self-check is a real `assert` and passes, at-risk list (3,604) has revenue-at-risk. ✅ verified 2026-07-14.
- [x] Segmentation: log1p applied before scaling, **k=5** justified (elbow + local silhouette max, k<3 excluded), **stability 100%**, silhouette 0.265, GMM cross-check, 4,320 rows in DuckDB. ✅ verified.
- [x] Forecasting: **AutoARIMA (1,0,1)(0,1,2)[7]**, 5-fold walk-forward **MAPE 27.9%**, beats seasonal-naive (32.2%) by **13.5%**, 95% intervals present. ✅ verified.
- [x] All three wrote `outputs/metrics/*.json` with real numbers. ✅
- [x] No new fake claims introduced (Prophet claim removed; honest caveats embedded in each JSON). ✅

**Phase 1 (models) COMPLETE 2026-07-14.** Next: Phase 2 LLM grounding guard → Phase 3 dbt → Phase 4 Next.js → Phase 5 resume rewrite.
