"""
RevenueIQ AI - Task 2: Data Cleaning Execution
Runs the complete cleaning pipeline
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from data_cleaning import clean_transaction_data, generate_cleaning_report
from datetime import datetime

# Paths
DATA_DIR = project_root / 'data'
RAW_DATA = DATA_DIR / 'raw' / 'Online Retail.xlsx'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = project_root / 'reports'

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("REVENUEIQ AI - TASK 2: DATA CLEANING & PREPROCESSING")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Step 1: Load raw data
print("📂 Loading raw data...")
df_raw = pd.read_excel(RAW_DATA)
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
print(f"✅ Loaded {len(df_raw):,} transactions")
print()

# Step 2: Clean data (keep returns flagged)
print("🧹 Running cleaning pipeline...")
df_cleaned, summary = clean_transaction_data(df_raw, remove_returns=False)
print()

# Step 3: Save cleaned data
print("💾 Saving cleaned data...")

# Full dataset with returns flagged
full_output = PROCESSED_DIR / 'transactions_cleaned.csv'
df_cleaned.to_csv(full_output, index=False)
print(f"✅ Saved full cleaned dataset: {full_output}")

# Also save Excel version
full_output_excel = PROCESSED_DIR / 'transactions_cleaned.xlsx'
df_cleaned.to_excel(full_output_excel, index=False)
print(f"✅ Saved Excel version: {full_output_excel}")

# Sales only (no returns) - for revenue analysis
df_sales = df_cleaned[~df_cleaned['IsReturn']].copy()
sales_output = PROCESSED_DIR / 'transactions_sales_only.csv'
df_sales.to_csv(sales_output, index=False)
print(f"✅ Saved sales-only dataset: {sales_output}")

print()

# Step 4: Generate report
print("📊 Generating cleaning report...")
report_path = REPORTS_DIR / 'task2_cleaning_report.txt'
generate_cleaning_report(summary, report_path)
print()

# Step 5: Display summary statistics
print("="*70)
print("CLEANED DATA SUMMARY")
print("="*70)
print(f"Total Transactions:     {len(df_cleaned):,}")
print(f"Sales Transactions:     {len(df_sales):,}")
print(f"Return Transactions:    {df_cleaned['IsReturn'].sum():,}")
print(f"Unique Customers:       {df_cleaned['CustomerID'].nunique():,}")
print(f"Unique Products:        {df_cleaned['StockCode'].nunique():,}")
print(f"Unique Invoices:        {df_cleaned['InvoiceNo'].nunique():,}")
print(f"Date Range:             {df_cleaned['InvoiceDate'].min()} to {df_cleaned['InvoiceDate'].max()}")
print()

# Revenue summary
total_revenue = df_sales['TotalPrice'].sum()
avg_transaction = df_sales.groupby('InvoiceNo')['TotalPrice'].sum().mean()

print("REVENUE METRICS (Sales Only)")
print("-"*70)
print(f"Total Revenue:          ${total_revenue:,.2f}")
print(f"Average Transaction:    ${avg_transaction:,.2f}")
print(f"Average Item Price:     ${df_sales['UnitPrice'].mean():.2f}")
print()

print("="*70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("✅ TASK 2 COMPLETE - Data cleaning successful!")
print("="*70)
