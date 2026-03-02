import pandas as pd
import sys

# Load data
df = pd.read_csv('data/processed/transactions_cleaned.csv')
df_sales = pd.read_csv('data/processed/transactions_sales_only.csv')

print("="*70)
print("REVENUEIQ AI - DATA VIEWER")
print("="*70)
print()

print("📊 DATASET INFO")
print("-"*70)
print(f"Total Transactions:     {len(df):,}")
print(f"Sales Only:             {len(df_sales):,}")
print(f"Returns:                {df['IsReturn'].sum():,}")
print()

print("📋 COLUMNS AVAILABLE")
print("-"*70)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
print()

print("🔍 SAMPLE DATA (First 10 Rows)")
print("-"*70)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head(10))
print()

print("📈 QUICK STATISTICS")
print("-"*70)
print(df[['Quantity', 'UnitPrice', 'TotalPrice']].describe())
print()

print("🔝 TOP 5 CUSTOMERS BY REVENUE")
print("-"*70)
top_customers = df_sales.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head()
for customer, revenue in top_customers.items():
    print(f"  {customer:20s} ${revenue:>12,.2f}")
print()

print("🛍️ TOP 5 PRODUCTS BY QUANTITY SOLD")
print("-"*70)
top_products = df_sales.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head()
for product, qty in top_products.items():
    print(f"  {product[:40]:40s} {qty:>8,} units")
print()

print("="*70)
