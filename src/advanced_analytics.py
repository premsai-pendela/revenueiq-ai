"""
RevenueIQ AI - Advanced Analytics Module
Task 3A: RFM Analysis, CLV, Cohort Analysis, Product Affinity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class AdvancedAnalytics:
    """Advanced customer and product analytics"""
    
    def __init__(self, df, output_dir='visualizations'):
        """
        Initialize Advanced Analytics
        
        Args:
            df: Cleaned transaction DataFrame
            output_dir: Directory for visualizations
        """
        self.df = df.copy()
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.insights = {}
        
        print(f"🔬 Advanced Analytics initialized with {len(df):,} transactions")
    
    def rfm_analysis(self, reference_date=None):
        """
        Perform RFM (Recency, Frequency, Monetary) Analysis
        
        Args:
            reference_date: Reference date for recency calculation (default: last date in data)
        """
        print("\n📊 Performing RFM Analysis...")
        
        # Set reference date (last date + 1 day)
        if reference_date is None:
            reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        
        # Filter out Guest customers for meaningful analysis
        df_registered = self.df[self.df['CustomerID'] != 'Guest'].copy()
        
        # Calculate RFM metrics
        rfm = df_registered.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores (1-5, 5 being best)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to int
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate RFM Score (combination)
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        # Segment customers
        def segment_customer(row):
            if row['RFM_Score'] >= 13:
                return 'Champions'
            elif row['RFM_Score'] >= 11:
                return 'Loyal Customers'
            elif row['RFM_Score'] >= 9:
                return 'Potential Loyalists'
            elif row['RFM_Score'] >= 7:
                return 'Recent Customers'
            elif row['RFM_Score'] >= 5:
                return 'At Risk'
            else:
                return 'Lost'
        
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        
        # Store insights
        segment_counts = rfm['Segment'].value_counts()
        segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        
        self.insights['rfm'] = {
            'total_customers_analyzed': len(rfm),
            'champions_count': segment_counts.get('Champions', 0),
            'champions_revenue': segment_revenue.get('Champions', 0),
            'at_risk_count': segment_counts.get('At Risk', 0),
            'lost_count': segment_counts.get('Lost', 0),
            'avg_recency': rfm['Recency'].mean(),
            'avg_frequency': rfm['Frequency'].mean(),
            'avg_monetary': rfm['Monetary'].mean()
        }
        
        # Save RFM data
        rfm.to_csv(self.output_dir.parent / 'data' / 'processed' / 'rfm_analysis.csv', index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RFM Analysis - Customer Segmentation', fontsize=18, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts.plot(kind='bar', ax=axes[0,0], color='steelblue')
        axes[0,0].set_title('Customer Count by Segment', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Segment')
        axes[0,0].set_ylabel('Number of Customers')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Revenue by segment
        segment_revenue.plot(kind='bar', ax=axes[0,1], color='green')
        axes[0,1].set_title('Revenue by Segment', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Segment')
        axes[0,1].set_ylabel('Revenue ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Recency vs Monetary
        scatter = axes[1,0].scatter(rfm['Recency'], rfm['Monetary'], 
                                    c=rfm['RFM_Score'], cmap='RdYlGn', alpha=0.6)
        axes[1,0].set_title('Recency vs Monetary Value', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Recency (days)')
        axes[1,0].set_ylabel('Monetary ($)')
        plt.colorbar(scatter, ax=axes[1,0], label='RFM Score')
        
        # 4. Frequency distribution
        rfm['Frequency'].hist(bins=30, ax=axes[1,1], color='orange', edgecolor='black')
        axes[1,1].set_title('Customer Purchase Frequency Distribution', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Number of Purchases')
        axes[1,1].set_ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rfm_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ RFM Analysis complete - {len(rfm):,} customers segmented")
        print(f"   Champions: {segment_counts.get('Champions', 0):,}")
        print(f"   At Risk: {segment_counts.get('At Risk', 0):,}")
        print(f"   Lost: {segment_counts.get('Lost', 0):,}")
        
        return rfm
    
    def customer_lifetime_value(self, months=12):
        """
        Calculate Customer Lifetime Value (CLV)
        
        Args:
            months: Period for calculation (default: 12 months)
        """
        print(f"\n💰 Calculating Customer Lifetime Value ({months} months)...")
        
        # Filter registered customers only
        df_registered = self.df[self.df['CustomerID'] != 'Guest'].copy()
        
        # Calculate per customer
        customer_stats = df_registered.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',  # Purchase frequency
            'TotalPrice': 'sum',  # Total spent
            'InvoiceDate': lambda x: (x.max() - x.min()).days  # Customer lifespan
        }).reset_index()
        
        customer_stats.columns = ['CustomerID', 'PurchaseCount', 'TotalRevenue', 'LifespanDays']
        
        # Calculate metrics
        customer_stats['AvgOrderValue'] = customer_stats['TotalRevenue'] / customer_stats['PurchaseCount']
        customer_stats['PurchaseFrequency'] = customer_stats['PurchaseCount'] / (customer_stats['LifespanDays'] / 30)  # per month
        
        # Simple CLV = Avg Order Value × Purchase Frequency × Customer Lifespan (in months)
        customer_stats['LifespanMonths'] = customer_stats['LifespanDays'] / 30
        customer_stats['CLV'] = (customer_stats['AvgOrderValue'] * 
                                 customer_stats['PurchaseFrequency'] * 
                                 customer_stats['LifespanMonths'])
        
        # Handle infinity and NaN
        customer_stats['CLV'] = customer_stats['CLV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Store insights
        self.insights['clv'] = {
            'avg_clv': customer_stats['CLV'].mean(),
            'median_clv': customer_stats['CLV'].median(),
            'top_clv': customer_stats['CLV'].max(),
            'avg_order_value': customer_stats['AvgOrderValue'].mean(),
            'avg_purchase_frequency': customer_stats['PurchaseFrequency'].mean(),
            'total_customer_value': customer_stats['CLV'].sum()
        }
        
        # Save CLV data
        customer_stats.to_csv(self.output_dir.parent / 'data' / 'processed' / 'customer_clv.csv', index=False)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Lifetime Value Analysis', fontsize=18, fontweight='bold')
        
        # 1. CLV Distribution
        customer_stats['CLV'].hist(bins=50, ax=axes[0,0], color='purple', edgecolor='black')
        axes[0,0].set_title('CLV Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Customer Lifetime Value ($)')
        axes[0,0].set_ylabel('Number of Customers')
        axes[0,0].axvline(customer_stats['CLV'].mean(), color='red', linestyle='--', label=f'Mean: ${customer_stats["CLV"].mean():.2f}')
        axes[0,0].legend()
        
        # 2. Top 20 customers by CLV
        top_20_clv = customer_stats.nlargest(20, 'CLV')
        axes[0,1].barh(range(20), top_20_clv['CLV'].values, color='green')
        axes[0,1].set_title('Top 20 Customers by CLV', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('CLV ($)')
        axes[0,1].set_ylabel('Customer Rank')
        axes[0,1].invert_yaxis()
        
        # 3. AOV vs Purchase Frequency
        scatter = axes[1,0].scatter(customer_stats['PurchaseFrequency'], 
                                    customer_stats['AvgOrderValue'],
                                    c=customer_stats['CLV'], cmap='viridis', alpha=0.6)
        axes[1,0].set_title('Average Order Value vs Purchase Frequency', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Purchase Frequency (per month)')
        axes[1,0].set_ylabel('Average Order Value ($)')
        plt.colorbar(scatter, ax=axes[1,0], label='CLV ($)')
        
        # 4. CLV segments
        clv_segments = pd.cut(customer_stats['CLV'], bins=[0, 100, 500, 1000, 5000, np.inf],
                             labels=['<$100', '$100-500', '$500-1K', '$1K-5K', '>$5K'])
        clv_segments.value_counts().sort_index().plot(kind='bar', ax=axes[1,1], color='teal')
        axes[1,1].set_title('Customer Segments by CLV', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('CLV Range')
        axes[1,1].set_ylabel('Number of Customers')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'customer_lifetime_value.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ CLV Analysis complete")
        print(f"   Average CLV: ${customer_stats['CLV'].mean():.2f}")
        print(f"   Median CLV: ${customer_stats['CLV'].median():.2f}")
        print(f"   Highest CLV: ${customer_stats['CLV'].max():.2f}")
        
        return customer_stats
    
    def cohort_analysis(self):
        """Perform cohort retention analysis"""
        print("\n📅 Performing Cohort Analysis...")
        
        # Filter registered customers
        df_registered = self.df[self.df['CustomerID'] != 'Guest'].copy()
        
        # Create cohort month (first purchase month)
        df_registered['InvoiceMonth'] = df_registered['InvoiceDate'].dt.to_period('M')
        df_registered['CohortMonth'] = df_registered.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
        
        # Calculate cohort index (months since first purchase)
        def get_month_int(period):
            return period.year * 12 + period.month
        
        df_registered['CohortIndex'] = df_registered['InvoiceMonth'].apply(get_month_int) - df_registered['CohortMonth'].apply(get_month_int)
        
        # Create cohort table
        cohort_data = df_registered.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
        cohort_data.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)
        
        # Pivot to create cohort matrix
        cohort_matrix = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerCount')
        
        # Calculate retention rate
        cohort_size = cohort_matrix.iloc[:, 0]
        retention_matrix = cohort_matrix.divide(cohort_size, axis=0) * 100
        
        # Store insights
        self.insights['cohort'] = {
            'total_cohorts': len(cohort_matrix),
            'avg_month1_retention': retention_matrix.iloc[:, 1].mean() if len(retention_matrix.columns) > 1 else 0,
            'avg_month3_retention': retention_matrix.iloc[:, 3].mean() if len(retention_matrix.columns) > 3 else 0,
            'best_cohort': cohort_size.idxmax(),
            'best_cohort_size': cohort_size.max()
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Cohort Retention Analysis', fontsize=18, fontweight='bold')
        
        # 1. Retention heatmap (percentage)
        sns.heatmap(retention_matrix, annot=True, fmt='.0f', cmap='RdYlGn', ax=axes[0], 
                    cbar_kws={'label': 'Retention %'})
        axes[0].set_title('Customer Retention Rate by Cohort (%)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Months Since First Purchase')
        axes[0].set_ylabel('Cohort (First Purchase Month)')
        
        # 2. Cohort sizes
        cohort_size.plot(kind='bar', ax=axes[1], color='steelblue')
        axes[1].set_title('Cohort Sizes (First Purchase Month)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Cohort Month')
        axes[1].set_ylabel('Number of Customers')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cohort_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cohort Analysis complete")
        print(f"   Total cohorts: {len(cohort_matrix)}")
        if len(retention_matrix.columns) > 1:
            print(f"   Avg Month 1 retention: {retention_matrix.iloc[:, 1].mean():.1f}%")
        
        return retention_matrix
    
    def product_affinity_analysis(self, min_support=0.01):
        """
        Analyze which products are frequently bought together
        
        Args:
            min_support: Minimum support for association rules (default: 1%)
        """
        print("\n🛒 Analyzing Product Affinity (Market Basket Analysis)...")
        
        # Create basket (invoice-product matrix)
        basket = self.df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
        
        # Convert to binary (bought or not)
        basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate product co-occurrence
        from itertools import combinations
        
        # Get top 50 products for analysis (to keep it manageable)
        top_products = self.df.groupby('Description')['Quantity'].sum().nlargest(50).index.tolist()
        
        # Find product pairs
        product_pairs = []
        
        for invoice in basket_binary.index:
            products_in_invoice = basket_binary.loc[invoice][basket_binary.loc[invoice] > 0].index.tolist()
            products_in_invoice = [p for p in products_in_invoice if p in top_products]
            
            if len(products_in_invoice) >= 2:
                for pair in combinations(products_in_invoice, 2):
                    product_pairs.append(pair)
        
        # Count pairs
        from collections import Counter
        pair_counts = Counter(product_pairs)
        
        # Convert to DataFrame
        affinity_df = pd.DataFrame(pair_counts.most_common(20), columns=['ProductPair', 'Count'])
        affinity_df['Product1'] = affinity_df['ProductPair'].apply(lambda x: x[0][:30])
        affinity_df['Product2'] = affinity_df['ProductPair'].apply(lambda x: x[1][:30])
        
        # Store insights
        self.insights['affinity'] = {
            'total_unique_pairs': len(pair_counts),
            'top_pair': affinity_df.iloc[0]['ProductPair'] if len(affinity_df) > 0 else None,
            'top_pair_count': affinity_df.iloc[0]['Count'] if len(affinity_df) > 0 else 0,
            'avg_basket_size': (basket_binary.sum(axis=1) > 0).sum() / len(basket_binary)
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        y_pos = range(len(affinity_df))
        labels = [f"{row['Product1']}\n + {row['Product2']}" for _, row in affinity_df.iterrows()]
        
        ax.barh(y_pos, affinity_df['Count'], color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Times Bought Together', fontsize=12)
        ax.set_title('Top 20 Product Pairs Frequently Bought Together', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'product_affinity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Product Affinity Analysis complete")
        print(f"   Analyzed {len(pair_counts):,} unique product pairs")
        if len(affinity_df) > 0:
            print(f"   Top pair bought together {affinity_df.iloc[0]['Count']:,} times")
        
        return affinity_df
    
    def generate_advanced_report(self, output_path):
        """Generate comprehensive advanced analytics report"""
        print("\n📝 Generating Advanced Analytics Report...")
        
        report_lines = [
            "="*70,
            "REVENUEIQ AI - ADVANCED ANALYTICS REPORT",
            "TASK 3A: Customer Segmentation & Product Insights",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*70,
            "",
        ]
        
        # RFM insights
        if 'rfm' in self.insights:
            rfm = self.insights['rfm']
            report_lines.extend([
                "📊 RFM ANALYSIS",
                "-"*70,
                f"Customers Analyzed:         {rfm['total_customers_analyzed']:,}",
                f"Champions:                  {rfm['champions_count']:,} (${rfm['champions_revenue']:,.2f})",
                f"At Risk:                    {rfm['at_risk_count']:,}",
                f"Lost Customers:             {rfm['lost_count']:,}",
                f"Avg Recency:                {rfm['avg_recency']:.0f} days",
                f"Avg Purchase Frequency:     {rfm['avg_frequency']:.1f} orders",
                f"Avg Customer Value:         ${rfm['avg_monetary']:,.2f}",
                "",
            ])
        
        # CLV insights
        if 'clv' in self.insights:
            clv = self.insights['clv']
            report_lines.extend([
                "💰 CUSTOMER LIFETIME VALUE",
                "-"*70,
                f"Average CLV:                ${clv['avg_clv']:,.2f}",
                f"Median CLV:                 ${clv['median_clv']:,.2f}",
                f"Highest CLV:                ${clv['top_clv']:,.2f}",
                f"Avg Order Value:            ${clv['avg_order_value']:,.2f}",
                f"Avg Purchase Frequency:     {clv['avg_purchase_frequency']:.2f} per month",
                f"Total Customer Value:       ${clv['total_customer_value']:,.2f}",
                "",
            ])
        
        # Cohort insights
        if 'cohort' in self.insights:
            cohort = self.insights['cohort']
            report_lines.extend([
                "📅 COHORT RETENTION ANALYSIS",
                "-"*70,
                f"Total Cohorts:              {cohort['total_cohorts']}",
                f"Avg Month 1 Retention:      {cohort['avg_month1_retention']:.1f}%",
                f"Avg Month 3 Retention:      {cohort['avg_month3_retention']:.1f}%",
                f"Largest Cohort:             {cohort['best_cohort']} ({cohort['best_cohort_size']} customers)",
                "",
            ])
        
        # Affinity insights
        if 'affinity' in self.insights:
            aff = self.insights['affinity']
            report_lines.extend([
                "🛒 PRODUCT AFFINITY ANALYSIS",
                "-"*70,
                f"Unique Product Pairs:       {aff['total_unique_pairs']:,}",
                f"Most Common Pair:           Bought together {aff['top_pair_count']:,} times",
                f"Avg Basket Size:            {aff['avg_basket_size']:.1f} items",
                "",
            ])
        
        report_lines.extend([
            "="*70,
            "STRATEGIC RECOMMENDATIONS",
            "-"*70,
            "1. Re-engage 'At Risk' customers with personalized offers",
            "2. Create VIP program for 'Champions' segment",
            "3. Focus retention efforts on high-CLV customers",
            "4. Bundle frequently co-purchased products",
            "5. Target month 1-3 with retention campaigns",
            "",
            "="*70,
            "ADVANCED ANALYSIS COMPLETE ✅",
            "="*70
        ])
        
        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Report saved to: {output_path}")
        print('\n'.join(report_lines))
        
        return self.insights


def run_advanced_analytics(df, output_dir='visualizations', 
                          report_path='reports/task3a_advanced_analytics.txt'):
    """
    Run all advanced analytics
    
    Args:
        df: Cleaned transaction DataFrame
        output_dir: Directory for visualizations
        report_path: Path for report
    
    Returns:
        Dictionary of insights
    """
    print("="*70)
    print("STARTING ADVANCED ANALYTICS")
    print("="*70)
    
    # Initialize
    analyzer = AdvancedAnalytics(df, output_dir)
    
    # Run analyses
    rfm_df = analyzer.rfm_analysis()
    clv_df = analyzer.customer_lifetime_value()
    cohort_matrix = analyzer.cohort_analysis()
    affinity_df = analyzer.product_affinity_analysis()
    
    # Generate report
    insights = analyzer.generate_advanced_report(report_path)
    
    print()
    print("="*70)
    print("ADVANCED ANALYTICS COMPLETE ✅")
    print("="*70)
    
    return insights, rfm_df, clv_df, cohort_matrix, affinity_df
