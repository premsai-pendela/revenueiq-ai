#!/usr/bin/env python3
"""
RevenueIQ AI - Day 1: Initial Data Exploration
Optimized for MacBook M4 Air
Date: March 1, 2026
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from data_loader import DataLoader

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

def print_header(text, char='='):
    """Print formatted header"""
    width = 70
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")

def print_subheader(text):
    """Print formatted subheader"""
    print(f"\n{'-' * 70}")
    print(f"  {text}")
    print(f"{'-' * 70}")

def main():
    """Main exploration function"""
    
    # Header
    print_header("REVENUEIQ AI - DAY 1: DATA EXPLORATION")
    print(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"💻 Platform: MacBook M4 Air")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print_header("[1] LOADING DATA", '-')
    
    loader = DataLoader('data/raw/Online Retail.xlsx')
    df = loader.load_data(verbose=True)
    
    if df is None:
        print("\n⚠️  Could not load data. Please check:")
        print("   1. File exists at: data/raw/Online Retail.xlsx")
        print("   2. You're in the project root directory")
        print(f"   3. Current directory: {Path.cwd()}")
        return
    
    # ==========================================
    # 2. DATASET OVERVIEW
    # ==========================================
    print_header("[2] DATASET OVERVIEW", '-')
    
    print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n📅 Date Range:")
    print(f"   Start: {df['InvoiceDate'].min()}")
    print(f"   End:   {df['InvoiceDate'].max()}")
    print(f"   Days:  {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days}")
    
    # ==========================================
    # 3. COLUMN INFORMATION
    # ==========================================
    print_header("[3] COLUMNS & DATA TYPES", '-')
    
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes.values,
        'Non_Null': df.notnull().sum().values,
        'Null_Count': df.isnull().sum().values,
        'Null_%': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    
    print(col_info.to_string(index=False))
    
    # ==========================================
    # 4. SAMPLE DATA
    # ==========================================
    print_header("[4] FIRST 5 RECORDS", '-')
    print(df.head().to_string())
    
    print_subheader("LAST 5 RECORDS")
    print(df.tail().to_string())
    
    # ==========================================
    # 5. MISSING VALUES ANALYSIS
    # ==========================================
    print_header("[5] MISSING VALUES ANALYSIS", '-')
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_%': missing_pct.values,
        'Present_Count': (len(df) - missing).values
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df.to_string(index=False))
    
    # ==========================================
    # 6. NUMERICAL STATISTICS
    # ==========================================
    print_header("[6] NUMERICAL COLUMNS STATISTICS", '-')
    print(df.describe().to_string())
    
    # ==========================================
    # 7. UNIQUE VALUES
    # ==========================================
    print_header("[7] UNIQUE VALUES ANALYSIS", '-')
    
    unique_df = pd.DataFrame({
        'Column': df.columns,
        'Unique_Count': [df[col].nunique() for col in df.columns],
        'Unique_%': [round(df[col].nunique() / len(df) * 100,2) for col in df.columns]
    })
    
    print(unique_df.to_string(index=False))
    
    # ==========================================
    # 8. TOP VALUES PER COLUMN
    # ==========================================
    print_header("[8] TOP VALUES (NON-NUMERIC COLUMNS)", '-')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print_subheader(f"Top 10 {col}")
        top_values = df[col].value_counts().head(10)
        print(top_values.to_string())
    
    # ==========================================
    # 9. DATA QUALITY ISSUES
    # ==========================================
    print_header("[9] DATA QUALITY ISSUES", '-')
    
    issues = []
    
    # Missing CustomerIDs
    missing_customers = df['CustomerID'].isnull().sum()
    missing_pct = (missing_customers / len(df) * 100)
    if missing_customers > 0:
        issues.append({
            'Issue': 'Missing CustomerID',
            'Count': missing_customers,
            'Percentage': f'{missing_pct:.2f}%',
            'Severity': 'HIGH' if missing_pct > 20 else 'MEDIUM'
        })
    
    # Negative quantities (returns/cancellations)
    negative_qty = (df['Quantity'] < 0).sum()
    if negative_qty > 0:
        issues.append({
            'Issue': 'Negative Quantity',
            'Count': negative_qty,
            'Percentage': f'{(negative_qty/len(df)*100):.2f}%',
            'Severity': 'MEDIUM'
        })
    
    # Zero or negative prices
    bad_prices = (df['UnitPrice'] <= 0).sum()
    if bad_prices > 0:
        issues.append({
            'Issue': 'Zero/Negative Price',
            'Count': bad_prices,
            'Percentage': f'{(bad_prices/len(df)*100):.2f}%',
            'Severity': 'HIGH'
        })
    
    # Missing descriptions
    missing_desc = df['Description'].isnull().sum()
    if missing_desc > 0:
        issues.append({
            'Issue': 'Missing Description',
            'Count': missing_desc,
            'Percentage': f'{(missing_desc/len(df)*100):.2f}%',
            'Severity': 'LOW'
        })
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            'Issue': 'Duplicate Rows',
            'Count': duplicates,
            'Percentage': f'{(duplicates/len(df)*100):.2f}%',
            'Severity': 'MEDIUM'
        })
    
    if issues:
        issues_df = pd.DataFrame(issues)
        print(issues_df.to_string(index=False))
    else:
        print("✅ No major data quality issues detected!")
    
    # ==========================================
    # 10. CANCELLATIONS ANALYSIS
    # ==========================================
    print_header("[10] CANCELLATIONS/RETURNS ANALYSIS", '-')
    
    # Invoices starting with 'C' are cancellations
    cancellations = df[df['InvoiceNo'].astype(str).str.startswith('C')]
    normal_invoices = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    print(f"Normal Invoices: {len(normal_invoices):,} ({len(normal_invoices)/len(df)*100:.2f}%)")
    print(f"Cancellations:   {len(cancellations):,} ({len(cancellations)/len(df)*100:.2f}%)")
    
    # ==========================================
    # 11. COUNTRY DISTRIBUTION
    # ==========================================
    print_header("[11] GEOGRAPHICAL DISTRIBUTION", '-')
    
    country_stats = df['Country'].value_counts()
    print(f"\nTotal Countries: {df['Country'].nunique()}")
    print(f"\nTop 10 Countries:")
    print(country_stats.head(10).to_string())
    
    # ==========================================
    # 12. SAVE SUMMARY REPORT
    # ==========================================
    print_header("[12] SAVING SUMMARY REPORT", '-')
    
    # Create outputs directory
    report_path = Path('outputs/reports')
    report_path.mkdir(parents=True, exist_ok=True)
    
    report_file = report_path / 'day1_exploration_summary.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REVENUEIQ AI - DAY 1 EXPLORATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}\n")
        f.write(f"Platform: MacBook M4 Air\n\n")
        
        f.write(f"DATASET OVERVIEW:\n")
        f.write(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"  Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}\n\n")
        
        f.write(f"COLUMNS:\n")
        f.write(col_info.to_string(index=False) + "\n\n")
        
        f.write(f"MISSING VALUES:\n")
        f.write(missing_df.to_string(index=False) + "\n\n")
        
        if issues:
            f.write(f"DATA QUALITY ISSUES:\n")
            f.write(issues_df.to_string(index=False) + "\n\n")
        
        f.write(f"KEY FINDINGS:\n")
        f.write(f"  • Total Transactions: {len(df):,}\n")
        f.write(f"  • Unique Customers: {df['CustomerID'].nunique():,}\n")
        f.write(f"  • Unique Products: {df['StockCode'].nunique():,}\n")
        f.write(f"  • Countries: {df['Country'].nunique()}\n")
        f.write(f"  • Cancellations: {len(cancellations):,} ({len(cancellations)/len(df)*100:.2f}%)\n")
    
    print(f"✅ Summary saved to: {report_file}")
    
    # ==========================================
    # COMPLETION
    # ==========================================
    print_header("DAY 1 EXPLORATION COMPLETE! 🎉", '=')
    
    print("📋 SUMMARY:")
    print(f"   • Dataset loaded: {df.shape[0]:,} transactions")
    print(f"   • Columns analyzed: {df.shape[1]}")
    print(f"   • Issues identified: {len(issues)}")
    print(f"   • Report saved: outputs/reports/day1_exploration_summary.txt")
    
    print("\n📝 NEXT STEPS (Day 2):")
    print("   1. Handle missing CustomerIDs")
    print("   2. Process cancellations/returns")
    print("   3. Clean price and quantity data")
    print("   4. Create calculated fields (TotalPrice, etc.)")
    print("   5. Set up DuckDB for efficient querying")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
