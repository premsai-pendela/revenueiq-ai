"""
RevenueIQ AI - Task 3A: Advanced Analytics Execution
RFM, CLV, Cohort, Product Affinity
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from advanced_analytics import run_advanced_analytics
from datetime import datetime

# Paths
DATA_DIR = project_root / 'data' / 'processed'
VIZ_DIR = project_root / 'visualizations'
REPORTS_DIR = project_root / 'reports'

print("="*70)
print("REVENUEIQ AI - TASK 3A: ADVANCED ANALYTICS")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load sales data (no returns)
print("📂 Loading sales data...")
df_sales = pd.read_csv(DATA_DIR / 'transactions_sales_only.csv')
print(f"✅ Loaded {len(df_sales):,} transactions")
print()

# Run advanced analytics
print("🔬 Running advanced analytics...")
insights, rfm_df, clv_df, cohort_matrix, affinity_df = run_advanced_analytics(
    df_sales,
    output_dir=str(VIZ_DIR),
    report_path=str(REPORTS_DIR / 'task3a_advanced_analytics.txt')
)
print()

# Summary
print("="*70)
print("NEW VISUALIZATIONS GENERATED")
print("="*70)
new_viz = ['rfm_analysis.png', 'customer_lifetime_value.png', 
           'cohort_analysis.png', 'product_affinity.png']
for viz in new_viz:
    viz_path = VIZ_DIR / viz
    if viz_path.exists():
        print(f"✅ {viz}")
print()

print("="*70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("✅ TASK 3A COMPLETE - Advanced insights generated!")
print("="*70)
