"""
RevenueIQ AI - SQL Analytics with DuckDB
Advanced SQL queries for business intelligence
"""

import duckdb
import pandas as pd
import time
from pathlib import Path

print("=" * 70)
print("REVENUEIQ - SQL ANALYTICS ENGINE")
print("Powered by DuckDB 🦆")
print("=" * 70)


class SQLAnalytics:
    """DuckDB-powered SQL analytics for revenue intelligence"""
    
    def __init__(self, db_path='data/revenueiq.db'):
        """Initialize DuckDB connection"""
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        print(f"\n✓ Connected to DuckDB: {db_path}")
    
    def load_data(self, csv_path='data/processed/transactions_cleaned.csv'):
        """Load CSV data into DuckDB table"""
        print(f"\n📊 Loading data from: {csv_path}")
        
        # Drop table if exists
        self.con.execute("DROP TABLE IF EXISTS transactions")
        
        # Create table from CSV
        self.con.execute(f"""
            CREATE TABLE transactions AS 
            SELECT * FROM read_csv_auto('{csv_path}')
        """)
        
        # Get row count
        count = self.con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        print(f"✓ Loaded {count:,} transactions into DuckDB")
        
        # Show table schema
        schema = self.con.execute("DESCRIBE transactions").df()
        print(f"\n📋 Table Schema:")
        print(schema.to_string(index=False))
        
        return count
    
    def query_1_revenue_by_country(self):
        """Top 10 countries by revenue - Basic aggregation"""
        print("\n" + "=" * 70)
        print("QUERY 1: Revenue by Country (GROUP BY)")
        print("=" * 70)
        
        query = """
            SELECT 
                Country,
                COUNT(DISTINCT InvoiceNo) as Orders,
                COUNT(DISTINCT CustomerID) as Customers,
                SUM(TotalPrice) as Revenue,
                AVG(TotalPrice) as AvgOrderValue,
                SUM(Quantity) as TotalUnits
            FROM transactions
            GROUP BY Country
            HAVING Revenue > 10000
            ORDER BY Revenue DESC
            LIMIT 10
        """
        
        result = self.con.execute(query).df()
        
        print("\n📊 Top 10 Countries by Revenue:")
        print(result.to_string(index=False))
        
        return result
    
    def query_2_monthly_trends(self):
        """Monthly revenue trends - Time series aggregation"""
        print("\n" + "=" * 70)
        print("QUERY 2: Monthly Revenue Trends")
        print("=" * 70)
        
        query = """
            SELECT 
                DATE_TRUNC('month', InvoiceDate) as Month,
                COUNT(DISTINCT InvoiceNo) as Orders,
                COUNT(DISTINCT CustomerID) as Customers,
                SUM(TotalPrice) as Revenue,
                AVG(TotalPrice) as AvgTransaction,
                SUM(Quantity) as Units
            FROM transactions
            GROUP BY DATE_TRUNC('month', InvoiceDate)
            ORDER BY Month
        """
        
        result = self.con.execute(query).df()
        
        print("\n📈 Monthly Performance:")
        print(result.to_string(index=False))
        
        return result
    
    def query_3_top_products(self):
        """Top selling products - Product analysis"""
        print("\n" + "=" * 70)
        print("QUERY 3: Top 15 Products by Revenue")
        print("=" * 70)
        
        query = """
            SELECT 
                Description as Product,
                COUNT(DISTINCT InvoiceNo) as TimesSold,
                SUM(Quantity) as UnitsSold,
                ROUND(AVG(UnitPrice), 2) as AvgPrice,
                ROUND(SUM(TotalPrice), 2) as TotalRevenue
            FROM transactions
            WHERE Description IS NOT NULL
                AND Description != 'POSTAGE'
            GROUP BY Description
            ORDER BY TotalRevenue DESC
            LIMIT 15
        """
        
        result = self.con.execute(query).df()
        
        print("\n🏆 Top 15 Products:")
        print(result.to_string(index=False))
        
        return result
    
    def query_4_customer_segments_sql(self):
        """RFM analysis in pure SQL - CTEs and Window Functions"""
        print("\n" + "=" * 70)
        print("QUERY 4: RFM Customer Segmentation (Advanced SQL)")
        print("=" * 70)
        
        query = """
            WITH customer_metrics AS (
                SELECT 
                    CustomerID,
                    MAX(InvoiceDate) as LastPurchase,
                    COUNT(DISTINCT InvoiceNo) as Frequency,
                    SUM(TotalPrice) as MonetaryValue
                FROM transactions
                WHERE CustomerID != 'Guest'
                GROUP BY CustomerID
            ),
            rfm_scores AS (
                SELECT 
                    CustomerID,
                    LastPurchase,
                    Frequency,
                    MonetaryValue,
                    -- Recency (days since last purchase)
                    DATEDIFF('day', LastPurchase, 
                        (SELECT MAX(InvoiceDate) FROM transactions)) as Recency,
                    -- RFM Scores (1-5 scale)
                    NTILE(5) OVER (ORDER BY DATEDIFF('day', LastPurchase, 
                        (SELECT MAX(InvoiceDate) FROM transactions)) DESC) as R_Score,
                    NTILE(5) OVER (ORDER BY Frequency) as F_Score,
                    NTILE(5) OVER (ORDER BY MonetaryValue) as M_Score
                FROM customer_metrics
            )
            SELECT 
                CASE 
                    WHEN R_Score >= 4 AND F_Score >= 4 THEN 'Champions'
                    WHEN R_Score >= 3 AND F_Score >= 3 THEN 'Loyal'
                    WHEN R_Score >= 3 AND F_Score < 3 THEN 'Potential'
                    WHEN R_Score < 3 AND F_Score >= 3 THEN 'At Risk'
                    ELSE 'Lost'
                END as Segment,
                COUNT(*) as Customers,
                ROUND(AVG(MonetaryValue), 2) as AvgRevenue,
                ROUND(SUM(MonetaryValue), 2) as TotalRevenue,
                ROUND(AVG(Frequency), 1) as AvgOrders,
                ROUND(AVG(Recency), 0) as AvgDaysSinceLast
            FROM rfm_scores
            GROUP BY Segment
            ORDER BY TotalRevenue DESC
        """
        
        result = self.con.execute(query).df()
        
        print("\n🎯 Customer Segments (SQL-based RFM):")
        print(result.to_string(index=False))
        
        return result
    
    def query_5_hourly_patterns(self):
        """Sales patterns by hour and day - Time analysis"""
        print("\n" + "=" * 70)
        print("QUERY 5: Sales Patterns by Time")
        print("=" * 70)
        
        query = """
            SELECT 
                EXTRACT(hour FROM InvoiceDate) as Hour,
                DAYNAME(InvoiceDate) as DayOfWeek,
                COUNT(DISTINCT InvoiceNo) as Orders,
                ROUND(SUM(TotalPrice), 2) as Revenue,
                ROUND(AVG(TotalPrice), 2) as AvgTransaction
            FROM transactions
            GROUP BY EXTRACT(hour FROM InvoiceDate), DAYNAME(InvoiceDate)
            ORDER BY Revenue DESC
            LIMIT 20
        """
        
        result = self.con.execute(query).df()
        
        print("\n⏰ Top 20 Peak Hours:")
        print(result.to_string(index=False))
        
        return result
    
    def query_6_customer_lifetime_value(self):
        """Calculate CLV with SQL window functions"""
        print("\n" + "=" * 70)
        print("QUERY 6: Top 20 Customers by Lifetime Value")
        print("=" * 70)
        
        query = """
            WITH customer_stats AS (
                SELECT 
                    CustomerID,
                    COUNT(DISTINCT InvoiceNo) as TotalOrders,
                    SUM(TotalPrice) as LifetimeValue,
                    AVG(TotalPrice) as AvgOrderValue,
                    MIN(InvoiceDate) as FirstPurchase,
                    MAX(InvoiceDate) as LastPurchase,
                    DATEDIFF('day', MIN(InvoiceDate), MAX(InvoiceDate)) as CustomerLifespanDays
                FROM transactions
                WHERE CustomerID != 'Guest'
                GROUP BY CustomerID
            )
            SELECT 
                CustomerID,
                TotalOrders,
                ROUND(LifetimeValue, 2) as LifetimeValue,
                ROUND(AvgOrderValue, 2) as AvgOrderValue,
                FirstPurchase,
                LastPurchase,
                CustomerLifespanDays,
                -- Rank customers
                ROW_NUMBER() OVER (ORDER BY LifetimeValue DESC) as Rank
            FROM customer_stats
            WHERE TotalOrders >= 5
            ORDER BY LifetimeValue DESC
            LIMIT 20
        """
        
        result = self.con.execute(query).df()
        
        print("\n💰 Top 20 Most Valuable Customers:")
        print(result.to_string(index=False))
        
        return result
    
    def query_7_product_affinity(self):
        """Product combinations - Self JOIN"""
        print("\n" + "=" * 70)
        print("QUERY 7: Product Affinity Analysis (Products Bought Together)")
        print("=" * 70)
        
        query = """
            SELECT 
                t1.Description as Product_A,
                t2.Description as Product_B,
                COUNT(DISTINCT t1.InvoiceNo) as TimesBoughtTogether,
                ROUND(SUM(t1.TotalPrice + t2.TotalPrice), 2) as CombinedRevenue
            FROM transactions t1
            JOIN transactions t2 
                ON t1.InvoiceNo = t2.InvoiceNo 
                AND t1.Description < t2.Description
            WHERE t1.Description IS NOT NULL 
                AND t2.Description IS NOT NULL
                AND t1.Description != 'POSTAGE'
                AND t2.Description != 'POSTAGE'
            GROUP BY Product_A, Product_B
            HAVING TimesBoughtTogether >= 50
            ORDER BY TimesBoughtTogether DESC
            LIMIT 15
        """
        
        result = self.con.execute(query).df()
        
        print("\n🔗 Top 15 Product Combinations:")
        print(result.to_string(index=False))
        
        return result
    
    def query_8_cohort_retention(self):
        """Cohort retention analysis - Advanced window functions"""
        print("\n" + "=" * 70)
        print("QUERY 8: Monthly Cohort Retention Analysis")
        print("=" * 70)
        
        query = """
            WITH first_purchase AS (
                SELECT 
                    CustomerID,
                    DATE_TRUNC('month', MIN(InvoiceDate)) as CohortMonth
                FROM transactions
                WHERE CustomerID != 'Guest'
                GROUP BY CustomerID
            ),
            purchase_months AS (
                SELECT DISTINCT
                    t.CustomerID,
                    fp.CohortMonth,
                    DATE_TRUNC('month', t.InvoiceDate) as PurchaseMonth,
                    DATEDIFF('month', fp.CohortMonth, 
                        DATE_TRUNC('month', t.InvoiceDate)) as MonthNumber
                FROM transactions t
                JOIN first_purchase fp ON t.CustomerID = fp.CustomerID
                WHERE t.CustomerID != 'Guest'
            )
            SELECT 
                CohortMonth,
                COUNT(DISTINCT CASE WHEN MonthNumber = 0 THEN CustomerID END) as Month_0,
                COUNT(DISTINCT CASE WHEN MonthNumber = 1 THEN CustomerID END) as Month_1,
                COUNT(DISTINCT CASE WHEN MonthNumber = 2 THEN CustomerID END) as Month_2,
                COUNT(DISTINCT CASE WHEN MonthNumber = 3 THEN CustomerID END) as Month_3,
                -- Retention rates
                ROUND(100.0 * COUNT(DISTINCT CASE WHEN MonthNumber = 1 THEN CustomerID END) / 
                    NULLIF(COUNT(DISTINCT CASE WHEN MonthNumber = 0 THEN CustomerID END), 0), 1) as Retention_M1,
                ROUND(100.0 * COUNT(DISTINCT CASE WHEN MonthNumber = 2 THEN CustomerID END) / 
                    NULLIF(COUNT(DISTINCT CASE WHEN MonthNumber = 0 THEN CustomerID END), 0), 1) as Retention_M2,
                ROUND(100.0 * COUNT(DISTINCT CASE WHEN MonthNumber = 3 THEN CustomerID END) / 
                    NULLIF(COUNT(DISTINCT CASE WHEN MonthNumber = 0 THEN CustomerID END), 0), 1) as Retention_M3
            FROM purchase_months
            GROUP BY CohortMonth
            ORDER BY CohortMonth
        """
        
        result = self.con.execute(query).df()
        
        print("\n📅 Cohort Retention (%):")
        print(result.to_string(index=False))
        
        return result
    
    def benchmark_sql_vs_pandas(self):
        """Compare SQL vs Pandas performance"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK: SQL vs Pandas")
        print("=" * 70)
        
        # Load data for Pandas
        df = pd.read_csv('data/processed/transactions_cleaned.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Test Query: Revenue by Country
        print("\n🏁 Test: Revenue by Country (GROUP BY)")
        
        # SQL timing
        start = time.time()
        sql_result = self.con.execute("""
            SELECT Country, SUM(TotalPrice) as Revenue
            FROM transactions
            GROUP BY Country
            ORDER BY Revenue DESC
        """).df()
        sql_time = time.time() - start
        
        # Pandas timing
        start = time.time()
        pandas_result = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
        pandas_time = time.time() - start
        
        print(f"\n⏱️  SQL Time:    {sql_time:.4f} seconds")
        print(f"⏱️  Pandas Time: {pandas_time:.4f} seconds")
        print(f"⚡ Speedup:     {pandas_time/sql_time:.2f}x faster with SQL!")
        
        # Test 2: Complex aggregation
        print("\n🏁 Test: Monthly Trends (Complex aggregation)")
        
        start = time.time()
        sql_result = self.con.execute("""
            SELECT 
                DATE_TRUNC('month', InvoiceDate) as Month,
                COUNT(*) as Orders,
                SUM(TotalPrice) as Revenue
            FROM transactions
            GROUP BY DATE_TRUNC('month', InvoiceDate)
        """).df()
        sql_time2 = time.time() - start
        
        start = time.time()
        pandas_result = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg({
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'
        })
        pandas_time2 = time.time() - start
        
        print(f"\n⏱️  SQL Time:    {sql_time2:.4f} seconds")
        print(f"⏱️  Pandas Time: {pandas_time2:.4f} seconds")
        print(f"⚡ Speedup:     {pandas_time2/sql_time2:.2f}x faster with SQL!")
        
        avg_speedup = (pandas_time/sql_time + pandas_time2/sql_time2) / 2
        print(f"\n📊 Average Speedup: {avg_speedup:.2f}x")
        
        return {
            'test1_speedup': pandas_time/sql_time,
            'test2_speedup': pandas_time2/sql_time2,
            'avg_speedup': avg_speedup
        }
    
    def generate_sql_report(self):
        """Generate comprehensive SQL analytics report"""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE SQL ANALYTICS REPORT")
        print("=" * 70)
        
        results = {}
        
        # Run all queries
        results['revenue_by_country'] = self.query_1_revenue_by_country()
        results['monthly_trends'] = self.query_2_monthly_trends()
        results['top_products'] = self.query_3_top_products()
        results['customer_segments'] = self.query_4_customer_segments_sql()
        results['hourly_patterns'] = self.query_5_hourly_patterns()
        results['top_customers'] = self.query_6_customer_lifetime_value()
        results['product_affinity'] = self.query_7_product_affinity()
        results['cohort_retention'] = self.query_8_cohort_retention()
        
        # Performance benchmark
        benchmark = self.benchmark_sql_vs_pandas()
        
        # Save results
        print("\n💾 Saving results to CSV...")
        for name, df in results.items():
            output_path = f'outputs/sql_{name}.csv'
            df.to_csv(output_path, index=False)
            print(f"   ✓ {output_path}")
        
        # Generate summary report
        report_path = 'outputs/sql_performance_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("REVENUEIQ SQL ANALYTICS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Database: {self.db_path}\n")
            f.write(f"Total Queries Executed: 8\n\n")
            f.write("PERFORMANCE BENCHMARKS:\n")
            f.write(f"  Average SQL Speedup: {benchmark['avg_speedup']:.2f}x faster than Pandas\n")
            f.write(f"  Test 1 (GROUP BY): {benchmark['test1_speedup']:.2f}x faster\n")
            f.write(f"  Test 2 (Time Series): {benchmark['test2_speedup']:.2f}x faster\n\n")
            f.write("SQL TECHNIQUES DEMONSTRATED:\n")
            f.write("  ✓ GROUP BY & HAVING clauses\n")
            f.write("  ✓ Common Table Expressions (CTEs)\n")
            f.write("  ✓ Window Functions (NTILE, ROW_NUMBER)\n")
            f.write("  ✓ Self JOINs\n")
            f.write("  ✓ Date/Time functions\n")
            f.write("  ✓ Aggregation functions\n")
            f.write("  ✓ Subqueries\n")
            f.write("  ✓ CASE statements\n")
        
        print(f"\n✓ Performance report: {report_path}")
        
        print("\n" + "=" * 70)
        print("✅ SQL ANALYTICS COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Generated Files:")
        print(f"   - Database: {self.db_path}")
        print(f"   - 8 CSV reports in outputs/")
        print(f"   - Performance report: {report_path}")
        
        return results
    
    def close(self):
        """Close database connection"""
        self.con.close()
        print("\n✓ Database connection closed")


def main():
    """Execute SQL analytics pipeline"""
    
    # Initialize
    sql = SQLAnalytics()
    
    # Load data
    sql.load_data()
    
    # Generate comprehensive report
    results = sql.generate_sql_report()
    
    # Close connection
    sql.close()
    
    print("\n🎉 SQL Analytics Session Complete!")
    print("\nNext: KMeans Clustering (30 min)")


if __name__ == "__main__":
    main()
