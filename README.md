# 🚀 RevenueIQ AI - Advanced Data Analytics & Business Intelligence Platform

> End-to-end data analytics project analyzing **534K+ retail transactions** worth **$10.6M** in revenue, featuring automated data cleaning, advanced customer segmentation, and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-orange.svg)
![Dash](https://img.shields.io/badge/Dash-2.0+-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📊 Project Overview

**RevenueIQ AI** transforms raw e-commerce data into actionable business insights through:

- ✅ Automated data cleaning pipeline (98.56% retention)
- ✅ Advanced customer analytics (RFM, CLV, Cohort Analysis)
- ✅ Interactive web dashboard with 5 real-time filters
- ✅ 10 professional visualizations
- 🔜 Predictive analytics (Task 4 - Coming Soon!)

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Total Revenue** | $10,616,955.56 |
| **Transactions Analyzed** | 534,117 |
| **Time Period** | Dec 2010 - Dec 2011 (1 year) |
| **Unique Customers** | 4,339 + Guest shoppers |
| **Markets** | 38 countries |
| **Return Rate** | 1.7% (vs 8-10% industry avg) ✅ |

---

## 💡 Critical Business Insights

### 🔴 Urgent Opportunities

| Finding | Impact | Potential Revenue |
|---------|--------|------------------|
| **80% customer drop-off** | Only 20.6% return after first purchase | +$630K/year with retention program |
| **Guest conversion gap** | Registered spend 73% more per order | +$126K/year from 10% conversion |
| **Champion concentration** | 933 customers = 70% of revenue | Protect $6.2M with VIP program |
| **UK dependency** | 84.5% revenue from one market | +$1.6M from international expansion |

### 🟢 Positive Indicators

| Indicator | Value | Significance |
|-----------|-------|--------------|
| **Low return rate** | 1.7% | Excellent product quality |
| **High transaction value** | $532 avg | Bulk/B2B purchasing |
| **Strong seasonality** | Nov peak: $1.5M | Clear Q4 opportunity |
| **Predictable patterns** | Thu @ 10 AM | Optimal marketing timing |

### 🎯 Top 5 Recommendations

1. **30-day onboarding email sequence** → Fix 80% churn
2. **VIP loyalty program** → Protect $6.2M Champions
3. **Checkout signup incentive** → Convert guest shoppers (+$126K)
4. **Geographic expansion** → Germany, France, Netherlands
5. **Product bundling** → 1,162 affinity pairs identified

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Dash (Plotly) |
| **Excel Handling** | OpenPyXL |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure
revenueiq-ai/
│
├── 📂 data/
│ ├── raw/ # Original data (not in repo)
│ │ └── online_retail.xlsx # 541,909 transactions
│ └── processed/ # Cleaned data (not in repo)
│ ├── transactions_cleaned.csv # 534,117 records
│ ├── transactions_sales_only.csv
│ ├── rfm_analysis.csv
│ └── customer_clv.csv
│
├── 📂 src/ # Core modules
│ ├── data_cleaning.py # DataCleaner class
│ ├── eda_analysis.py # EDA functions
│ ├── advanced_analytics.py # RFM, CLV, Cohort
│ └── dashboard.py # Interactive dashboard
│
├── 📂 scripts/ # Execution scripts
│ ├── task1_load_data.py
│ ├── task2_clean_data.py
│ ├── task3_exploratory_analysis.py
│ ├── task3a_advanced_analytics.py
│ └── launch_dashboard.py
│
├── 📂 reports/ # Generated reports
│ ├── task1_initial_analysis.txt
│ ├── task2_cleaning_report.txt
│ ├── task3_eda_report.txt
│ └── task3a_advanced_analytics.txt
│
├── 📂 visualizations/ # Charts (not in repo)
│ ├── revenue_trends.png
│ ├── top_products.png
│ ├── customer_analysis.png
│ ├── geographic_analysis.png
│ ├── time_patterns.png
│ ├── executive_dashboard.png
│ ├── rfm_analysis.png
│ ├── customer_lifetime_value.png
│ ├── cohort_analysis.png
│ └── product_affinity.png
│
├── .gitignore
└── README.md


---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone repository
git clone https://github.com/premsai-pendela/revenueiq-ai.git
cd revenueiq-ai

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly dash openpyxl

Run the Analysis

# Step 1: Clean data
python scripts/task2_clean_data.py

# Step 2: Generate EDA
python scripts/task3_exploratory_analysis.py

# Step 3: Advanced analytics
python scripts/task3a_advanced_analytics.py

# Step 4: Launch dashboard
python scripts/launch_dashboard.py
# Then open: http://127.0.0.1:8050

## 📊 Analysis Results

### Task 1: Data Exploration

| Metric | Value |
|--------|-------|
| **Raw Transactions** | 541,909 |
| **Date Range** | Dec 2010 - Dec 2011 |
| **Original Columns** | 8 |

**Issues Found:**
- Missing CustomerID: 135,080 (24.93%)
- Negative Quantities: 10,624 (returns)
- Invalid Prices: 2,527
- Duplicates: 5,265

---

### Task 2: Data Cleaning

| Operation | Count | Action |
|-----------|-------|--------|
| Missing CustomerID | 135,080 | Labeled "Guest" |
| Invalid Prices | 2,527 | Removed |
| Duplicates | 5,265 | Removed |
| Returns | 10,624 | Flagged |

**Result:** 534,117 clean records (98.56% retention)

**New Fields Created:**
- TotalPrice (Quantity × UnitPrice)
- Year, Month, Day, Hour
- DayOfWeek, DayName, MonthName
- YearMonth, IsReturn

---

### Task 3: Exploratory Data Analysis

#### 💰 Revenue Insights

| Metric | Value |
|--------|-------|
| **Total Revenue** | $10,616,955.56 |
| **Daily Average** | $34,809.69 |
| **Best Day** | $200,900.98 |
| **Best Month** | November 2011 ($1.5M) |
| **Worst Month** | February 2011 ($522K) |

#### 🛍️ Product Insights

| Metric | Value |
|--------|-------|
| **Unique Products** | 3,921 |
| **Top by Revenue** | DOTCOM POSTAGE ($206K) |
| **Top by Quantity** | PAPER CRAFT BIRDIE (81K units) |

#### 👥 Customer Insights

| Segment | Transactions | Revenue | Avg Order |
|---------|--------------|---------|-----------|
| **Guest** | 132,184 (25%) | $1.73M (16%) | $13.09 |
| **Registered** | 392,690 (75%) | $8.89M (84%) | $22.63 |

**Key Insight:** Registered customers spend **73% MORE** per transaction!

#### 🌍 Geographic Insights

| Country | Revenue | Share |
|---------|---------|-------|
| United Kingdom | $8,976,588 | 84.5% |
| Netherlands | $284,662 | 2.7% |
| EIRE | $263,276 | 2.5% |
| Germany | $228,867 | 2.2% |
| France | $197,403 | 1.9% |

#### ⏰ Time Patterns

| Pattern | Peak | Low |
|---------|------|-----|
| **Hour** | 10 AM | 6 AM |
| **Day** | Thursday | Sunday |
| **Month** | November | February |

---

### Task 3A: Advanced Analytics

#### 📊 RFM Segmentation

| Segment | Count | Revenue | % of Revenue |
|---------|-------|---------|--------------|
| **Champions** | 933 | $6,241,008 | 70% |
| **Loyal** | 752 | $1,423,654 | 16% |
| **Potential Loyalists** | 891 | $612,445 | 7% |
| **At Risk** | 763 | $398,231 | 4% |
| **Lost** | 544 | $211,320 | 2% |

#### 💰 Customer Lifetime Value

| Metric | Value |
|--------|-------|
| **Average CLV** | $1,884.02 |
| **Median CLV** | $584.29 |
| **Highest CLV** | $280,206.02 |

#### 📅 Cohort Retention

| Cohort | Month 1 | Month 2 | Month 3 |
|--------|---------|---------|---------|
| Dec 2010 | 20.6% | 18.2% | 23.2% |
| Jan 2011 | 25.3% | 21.1% | 18.9% |

**Critical:** 80% of customers drop off after first purchase!

#### 🛒 Product Affinity

| Rank | Product Pair | Times Together |
|------|--------------|----------------|
| 1 | WHITE + CREAM HANGING HEART | 825 |
| 2 | JUMBO BAG RED + PINK | 612 |
| 3 | LUNCH BAG SUKI + RED | 543 |

---

## 🎨 Dashboard Features

### Interactive Filters

| Filter | Description |
|--------|-------------|
| 📅 Date Range | Custom time periods |
| 🌍 Country | 38 markets available |
| 🛍️ Product | Top 50 products |
| 👥 Customer Type | All / Guest / Registered |
| 📊 Time View | Daily / Weekly / Monthly |

### Dynamic Charts (7)

1. Revenue Trend Line
2. Top Products Bar
3. Geographic Pie
4. Hourly Pattern Bar
5. Customer Segments Bar
6. Day of Week Bar
7. KPI Cards (auto-update)

## 🎯 Strategic Recommendations

### 🔴 Tier 1: Urgent (30 Days)

**1. Fix Customer Retention Crisis**
- Problem: 80% churn after first purchase
- Solution: 30-day onboarding email sequence
- Impact: +$630K/year

**2. Protect High-Value Customers**
- Problem: 933 Champions = 70% revenue
- Solution: VIP loyalty program
- Impact: Protect $6.2M

**3. Convert Guest Customers**
- Problem: Guests spend 73% less
- Solution: Checkout signup incentive
- Impact: +$126K/year

### 🟡 Tier 2: Medium Priority (60-90 Days)

**4. Geographic Expansion**
- Target: Germany, France, Netherlands
- Current International: 15.5%
- Goal: 30%

**5. Marketing Optimization**
- Best time: Thursday @ 10 AM
- Action: Schedule campaigns accordingly

**6. Product Bundling**
- 1,162 affinity pairs identified
- Action: Create bundle deals

---

## 📈 Project Roadmap

| Phase | Task | Status |
|-------|------|--------|
| 1 | Data Loading & Exploration | ✅ Complete |
| 2 | Data Cleaning & Preprocessing | ✅ Complete |
| 3 | Exploratory Data Analysis | ✅ Complete |
| 3A | Advanced Analytics (RFM, CLV, Cohort) | ✅ Complete |
| 3C | Interactive Dashboard | ✅ Complete |
| 4 | Sales Forecasting (Time Series) | 🔜 Next |
| 5 | Customer Churn Prediction (ML) | 📋 Planned |
| 6 | Product Recommendation Engine | 📋 Planned |

---

## 🏆 Skills Demonstrated

### Technical

- Python (Pandas, NumPy)
- Data Cleaning & Validation
- Statistical Analysis
- Data Visualization (Matplotlib, Seaborn, Plotly)
- Web Dashboard Development (Dash)
- Git & GitHub

### Business

- Customer Segmentation (RFM)
- Lifetime Value Modeling (CLV)
- Cohort Retention Analysis
- Product Affinity Analysis
- Strategic Recommendations
- KPI Definition & Tracking

---

## 🐛 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| 1-year data only | Limited trends | Focus on seasonality |
| UK-heavy (84.5%) | Biased insights | Segment analysis |
| 25% guest transactions | Incomplete tracking | Separate analysis |
| Static data | No real-time | Manual refresh |
| No cost data | No profit margins | Revenue focus |

---

## 📚 References

**Dataset:**
- [UCI ML Repository - Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Dr. Daqing Chen, Brunel University London

**Documentation:**
- [Pandas](https://pandas.pydata.org/docs/)
- [Plotly Dash](https://dash.plotly.com/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 👤 Author

**Naga Prem Sai Pendela**

| | |
|---|---|
| **Role** | Data Analyst & BI Developer |
| **Focus** | E-commerce Analytics, Customer Segmentation |
| **GitHub** | [@premsai-pendela](https://github.com/premsai-pendela) |

---

## 📄 License

This project is for **educational and portfolio purposes**.

- Dataset: UCI ML Repository (Public)
- Code: Free to use and modify

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository
- Dr. Daqing Chen (Brunel University)
- Python Open Source Community

---

## ⭐ Support

**Found this helpful?**
- ⭐ Star the repository
- 🍴 Fork and build upon it
- 📝 Open issues for questions

---

**Last Updated:** March 2, 2026

**Status:** ✅ Active Development

---

<p align="center">
  <b>Built with ❤️ and ☕ by Naga Prem Sai Pendela</b>
</p>
