"""
RevenueIQ AI - Task 3: Exploratory Data Analysis Execution
Runs comprehensive EDA and generates visualizations
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from eda_analysis import run_full_eda
from datetime import datetime

# Paths
DATA_DIR = project_root / 'data' / 'processed'
VIZ_DIR = project_root / 'visualizations'
REPORTS_DIR = project_root / 'reports'

# Create directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("REVENUEIQ AI - TASK 3: EXPLORATORY DATA ANALYSIS")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load cleaned data
print("📂 Loading cleaned data...")
df = pd.read_csv(DATA_DIR / 'transactions_cleaned.csv')
df_sales = pd.read_csv(DATA_DIR / 'transactions_sales_only.csv')

print(f"✅ Loaded {len(df):,} transactions")
print(f"✅ Sales only: {len(df_sales):,} transactions")
print()

# Run EDA on sales data (excluding returns for cleaner analysis)
print("🔍 Running exploratory data analysis...")
insights = run_full_eda(
    df_sales, 
    output_dir=str(VIZ_DIR),
    report_path=str(REPORTS_DIR / 'task3_eda_report.txt')
)
print()

# Summary
print("="*70)
print("VISUALIZATIONS GENERATED")
print("="*70)
viz_files = list(VIZ_DIR.glob('*.png'))
for viz_file in sorted(viz_files):
    print(f"✅ {viz_file.name}")
print()

print("="*70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("✅ TASK 3 COMPLETE - Check visualizations/ folder!")
print("="*70)
