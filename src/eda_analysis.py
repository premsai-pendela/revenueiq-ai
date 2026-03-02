"""
RevenueIQ AI - Exploratory Data Analysis Module
Task 3: Comprehensive data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """Performs exploratory data analysis on transaction data"""
    
    def __init__(self, df, output_dir='visualizations'):
        """
        Initialize EDA Analyzer
        
        Args:
            df: pandas DataFrame with cleaned transaction data
            output_dir: Directory to save visualizations
        """
        self.df = df.copy()
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.insights = {}
        
        print(f"📊 EDA Analyzer initialized with {len(df):,} transactions")
    
    def analyze_revenue_trends(self):
        """Analyze revenue trends over time"""
        print("\n📈 Analyzing revenue trends...")
        
        # Daily revenue
        daily_revenue = self.df.groupby(self.df['InvoiceDate'].dt.date)['TotalPrice'].sum()
        
        # Monthly revenue
        monthly_revenue = self.df.groupby('YearMonth')['TotalPrice'].sum()
        
        # Weekly revenue
        weekly_revenue = self.df.groupby(self.df['InvoiceDate'].dt.to_period('W'))['TotalPrice'].sum()
        
        # Store insights
        self.insights['revenue'] = {
            'total': self.df['TotalPrice'].sum(),
            'daily_avg': daily_revenue.mean(),
            'daily_max': daily_revenue.max(),
            'daily_min': daily_revenue.min(),
            'monthly_avg': monthly_revenue.mean(),
            'best_month': monthly_revenue.idxmax(),
            'best_month_revenue': monthly_revenue.max(),
            'worst_month': monthly_revenue.idxmin(),
            'worst_month_revenue': monthly_revenue.min()
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily revenue trend
        axes[0].plot(daily_revenue.index, daily_revenue.values, linewidth=1, alpha=0.7)
        axes[0].set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Revenue ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Monthly revenue
        monthly_revenue.plot(kind='bar', ax=axes[1], color='steelblue')
        axes[1].set_title('Monthly Revenue', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Revenue ($)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'revenue_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Revenue trends analyzed and saved")
        return self.insights['revenue']
    
    def analyze_products(self, top_n=20):
        """Analyze product performance"""
        print(f"\n🛍️ Analyzing top {top_n} products...")
        
        # Top products by revenue
        product_revenue = self.df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)
        top_products_revenue = product_revenue.head(top_n)
        
        # Top products by quantity
        product_quantity = self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
        top_products_quantity = product_quantity.head(top_n)
        
        # Product statistics
        self.insights['products'] = {
            'total_unique': self.df['StockCode'].nunique(),
            'top_product': product_revenue.index[0],
            'top_product_revenue': product_revenue.iloc[0],
            'avg_product_revenue': product_revenue.mean(),
            'top_quantity_product': product_quantity.index[0],
            'top_quantity': product_quantity.iloc[0]
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top products by revenue
        top_products_revenue.plot(kind='barh', ax=axes[0], color='green')
        axes[0].set_title(f'Top {top_n} Products by Revenue', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Revenue ($)')
        axes[0].invert_yaxis()
        
        # Top products by quantity
        top_products_quantity.plot(kind='barh', ax=axes[1], color='orange')
        axes[1].set_title(f'Top {top_n} Products by Quantity Sold', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Quantity Sold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_products.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Product analysis completed")
        return self.insights['products']
    
    def analyze_customers(self, top_n=20):
        """Analyze customer behavior"""
        print(f"\n👥 Analyzing customer behavior...")
        
        # Customer revenue
        customer_revenue = self.df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
        top_customers = customer_revenue.head(top_n)
        
        # Customer frequency
        customer_frequency = self.df.groupby('CustomerID')['InvoiceNo'].nunique()
        
        # Guest vs Registered
        guest_count = (self.df['CustomerID'] == 'Guest').sum()
        registered_count = (self.df['CustomerID'] != 'Guest').sum()
        guest_revenue = self.df[self.df['CustomerID'] == 'Guest']['TotalPrice'].sum()
        registered_revenue = self.df[self.df['CustomerID'] != 'Guest']['TotalPrice'].sum()
        
        self.insights['customers'] = {
            'total_unique': self.df['CustomerID'].nunique(),
            'guest_transactions': guest_count,
            'registered_transactions': registered_count,
            'guest_revenue': guest_revenue,
            'registered_revenue': registered_revenue,
            'top_customer': customer_revenue.index[0],
            'top_customer_revenue': customer_revenue.iloc[0],
            'avg_customer_revenue': customer_revenue.mean(),
            'avg_purchase_frequency': customer_frequency.mean()
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top customers
        top_customers.plot(kind='barh', ax=axes[0], color='purple')
        axes[0].set_title(f'Top {top_n} Customers by Revenue', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Revenue ($)')
        axes[0].invert_yaxis()
        
        # Guest vs Registered
        segments = ['Guest', 'Registered']
        revenues = [guest_revenue, registered_revenue]
        axes[1].bar(segments, revenues, color=['coral', 'teal'])
        axes[1].set_title('Revenue: Guest vs Registered Customers', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Revenue ($)')
        
        # Add values on bars
        for i, v in enumerate(revenues):
            axes[1].text(i, v, f'${v:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'customer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Customer analysis completed")
        return self.insights['customers']
    
    def analyze_geography(self, top_n=10):
        """Analyze sales by geography"""
        print(f"\n🌍 Analyzing geographic distribution...")
        
        # Revenue by country
        country_revenue = self.df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
        top_countries = country_revenue.head(top_n)
        
        # Transaction count by country
        country_transactions = self.df.groupby('Country').size().sort_values(ascending=False)
        
        self.insights['geography'] = {
            'total_countries': self.df['Country'].nunique(),
            'top_country': country_revenue.index[0],
            'top_country_revenue': country_revenue.iloc[0],
            'top_country_share': (country_revenue.iloc[0] / country_revenue.sum() * 100),
            'international_revenue': country_revenue[country_revenue.index != 'United Kingdom'].sum()
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top countries by revenue
        top_countries.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title(f'Top {top_n} Countries by Revenue', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Revenue ($)')
        axes[0].invert_yaxis()
        
        # Pie chart for top 5
        top_5_countries = country_revenue.head(5)
        others = country_revenue[5:].sum()
        pie_data = list(top_5_countries.values) + [others]
        pie_labels = list(top_5_countries.index) + ['Others']
        
        axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Revenue Share by Country (Top 5 + Others)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Geographic analysis completed")
        return self.insights['geography']
    
    def analyze_time_patterns(self):
        """Analyze time-based patterns"""
        print("\n⏰ Analyzing time-based patterns...")
        
        # Hourly patterns
        hourly_sales = self.df.groupby('Hour')['TotalPrice'].sum()
        hourly_transactions = self.df.groupby('Hour').size()
        
        # Day of week patterns
        daily_sales = self.df.groupby('DayName')['TotalPrice'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales = daily_sales.reindex(day_order)
        
        # Best and worst hours
        peak_hour = hourly_sales.idxmax()
        slowest_hour = hourly_sales.idxmin()
        
        self.insights['time_patterns'] = {
            'peak_hour': int(peak_hour),
            'peak_hour_revenue': hourly_sales.max(),
            'slowest_hour': int(slowest_hour),
            'best_day': daily_sales.idxmax(),
            'best_day_revenue': daily_sales.max(),
            'worst_day': daily_sales.idxmin()
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Hourly pattern
        axes[0].plot(hourly_sales.index, hourly_sales.values, marker='o', linewidth=2, markersize=8)
        axes[0].set_title('Revenue by Hour of Day', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Revenue ($)')
        axes[0].set_xticks(range(24))
        axes[0].grid(True, alpha=0.3)
        
        # Day of week pattern
        daily_sales.plot(kind='bar', ax=axes[1], color='teal')
        axes[1].set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('Revenue ($)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Time pattern analysis completed")
        return self.insights['time_patterns']
    
    def create_summary_dashboard(self):
        """Create executive summary dashboard"""
        print("\n📊 Creating executive summary dashboard...")
        
        # Calculate key metrics
        total_revenue = self.df['TotalPrice'].sum()
        total_transactions = len(self.df)
        total_customers = self.df['CustomerID'].nunique()
        total_products = self.df['StockCode'].nunique()
        avg_transaction = self.df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        
        # Create dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('RevenueIQ AI - Executive Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # KPI boxes (top row)
        kpis = [
            ('Total Revenue', f'${total_revenue:,.0f}', 'green'),
            ('Transactions', f'{total_transactions:,}', 'blue'),
            ('Customers', f'{total_customers:,}', 'purple'),
        ]
        
        for idx, (label, value, color) in enumerate(kpis):
            ax = fig.add_subplot(gs[0, idx])
            ax.text(0.5, 0.5, value, ha='center', va='center', fontsize=24, fontweight='bold', color=color)
            ax.text(0.5, 0.2, label, ha='center', va='center', fontsize=12)
            ax.axis('off')
            ax.set_facecolor('#f0f0f0')
        
        # Revenue trend (middle left)
        ax1 = fig.add_subplot(gs[1, :2])
        monthly_revenue = self.df.groupby('YearMonth')['TotalPrice'].sum()
        ax1.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=2)
        ax1.set_title('Monthly Revenue Trend', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Revenue ($)')
        ax1.grid(True, alpha=0.3)
        
        # Top 5 products (middle right)
        ax2 = fig.add_subplot(gs[1, 2])
        top_5_products = self.df.groupby('Description')['TotalPrice'].sum().nlargest(5)
        top_5_products.plot(kind='barh', ax=ax2, color='orange')
        ax2.set_title('Top 5 Products', fontweight='bold')
        ax2.invert_yaxis()
        
        # Geographic split (bottom left)
        ax3 = fig.add_subplot(gs[2, 0])
        country_revenue = self.df.groupby('Country')['TotalPrice'].sum().nlargest(5)
        ax3.pie(country_revenue.values, labels=country_revenue.index, autopct='%1.1f%%')
        ax3.set_title('Top 5 Markets', fontweight='bold')
        
        # Hourly pattern (bottom middle)
        ax4 = fig.add_subplot(gs[2, 1])
        hourly = self.df.groupby('Hour')['TotalPrice'].sum()
        ax4.bar(hourly.index, hourly.values, color='teal')
        ax4.set_title('Sales by Hour', fontweight='bold')
        ax4.set_xlabel('Hour')
        
        # Customer segments (bottom right)
        ax5 = fig.add_subplot(gs[2, 2])
        guest_rev = self.df[self.df['CustomerID'] == 'Guest']['TotalPrice'].sum()
        reg_rev = self.df[self.df['CustomerID'] != 'Guest']['TotalPrice'].sum()
        ax5.bar(['Guest', 'Registered'], [guest_rev, reg_rev], color=['coral', 'steelblue'])
        ax5.set_title('Customer Segments', fontweight='bold')
        ax5.set_ylabel('Revenue ($)')
        
        plt.savefig(self.output_dir / 'executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Executive dashboard created")
    
    def generate_insights_report(self, output_path):
        """Generate comprehensive insights report"""
        print("\n📝 Generating insights report...")
        
        report_lines = [
            "="*70,
            "REVENUEIQ AI - EXPLORATORY DATA ANALYSIS REPORT",
            "TASK 3: Data Insights & Patterns",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*70,
            "",
        ]
        
        # Revenue insights
        if 'revenue' in self.insights:
            rev = self.insights['revenue']
            report_lines.extend([
                "💰 REVENUE INSIGHTS",
                "-"*70,
                f"Total Revenue:              ${rev['total']:,.2f}",
                f"Average Daily Revenue:      ${rev['daily_avg']:,.2f}",
                f"Best Day Revenue:           ${rev['daily_max']:,.2f}",
                f"Average Monthly Revenue:    ${rev['monthly_avg']:,.2f}",
                f"Best Month:                 {rev['best_month']} (${rev['best_month_revenue']:,.2f})",
                f"Worst Month:                {rev['worst_month']} (${rev['worst_month_revenue']:,.2f})",
                "",
            ])
        
        # Product insights
        if 'products' in self.insights:
            prod = self.insights['products']
            report_lines.extend([
                "🛍️ PRODUCT INSIGHTS",
                "-"*70,
                f"Total Unique Products:      {prod['total_unique']:,}",
                f"Top Revenue Product:        {prod['top_product'][:50]}",
                f"  Revenue:                  ${prod['top_product_revenue']:,.2f}",
                f"Top Quantity Product:       {prod['top_quantity_product'][:50]}",
                f"  Quantity Sold:            {prod['top_quantity']:,.0f} units",
                "",
            ])
        
        # Customer insights
        if 'customers' in self.insights:
            cust = self.insights['customers']
            report_lines.extend([
                "👥 CUSTOMER INSIGHTS",
                "-"*70,
                f"Total Unique Customers:     {cust['total_unique']:,}",
                f"Guest Transactions:         {cust['guest_transactions']:,}",
                f"  Revenue:                  ${cust['guest_revenue']:,.2f}",
                f"Registered Transactions:    {cust['registered_transactions']:,}",
                f"  Revenue:                  ${cust['registered_revenue']:,.2f}",
                f"Top Customer ID:            {cust['top_customer']}",
                f"  Revenue:                  ${cust['top_customer_revenue']:,.2f}",
                f"Avg Purchase Frequency:     {cust['avg_purchase_frequency']:.1f} orders",
                "",
            ])
        
        # Geographic insights
        if 'geography' in self.insights:
            geo = self.insights['geography']
            report_lines.extend([
                "🌍 GEOGRAPHIC INSIGHTS",
                "-"*70,
                f"Total Countries Served:     {geo['total_countries']}",
                f"Top Market:                 {geo['top_country']}",
                f"  Revenue:                  ${geo['top_country_revenue']:,.2f}",
                f"  Market Share:             {geo['top_country_share']:.1f}%",
                f"International Revenue:      ${geo['international_revenue']:,.2f}",
                "",
            ])
        
        # Time pattern insights
        if 'time_patterns' in self.insights:
            time = self.insights['time_patterns']
            report_lines.extend([
                "⏰ TIME PATTERN INSIGHTS",
                "-"*70,
                f"Peak Shopping Hour:         {time['peak_hour']}:00",
                f"  Revenue:                  ${time['peak_hour_revenue']:,.2f}",
                f"Slowest Hour:               {time['slowest_hour']}:00",
                f"Best Day of Week:           {time['best_day']}",
                f"  Revenue:                  ${time['best_day_revenue']:,.2f}",
                f"Slowest Day:                {time['worst_day']}",
                "",
            ])
        
        report_lines.extend([
            "="*70,
            "KEY TAKEAWAYS",
            "-"*70,
            "1. Revenue is concentrated in a few key products and customers",
            "2. Guest customers represent significant untapped potential",
            "3. Geographic expansion opportunities exist beyond UK",
            "4. Clear time patterns suggest optimal marketing windows",
            "5. Product portfolio shows high concentration - diversification opportunity",
            "",
            "="*70,
            "ANALYSIS COMPLETE ✅",
            "="*70
        ])
        
        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Report saved to: {output_path}")
        
        # Print to console
        print('\n'.join(report_lines))
        
        return self.insights


def run_full_eda(df, output_dir='visualizations', report_path='reports/task3_eda_report.txt'):
    """
    Run complete exploratory data analysis
    
    Args:
        df: Cleaned transaction DataFrame
        output_dir: Directory for visualizations
        report_path: Path for insights report
    
    Returns:
        Dictionary of all insights
    """
    print("="*70)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = EDAAnalyzer(df, output_dir)
    
    # Run all analyses
    analyzer.analyze_revenue_trends()
    analyzer.analyze_products(top_n=20)
    analyzer.analyze_customers(top_n=20)
    analyzer.analyze_geography(top_n=10)
    analyzer.analyze_time_patterns()
    analyzer.create_summary_dashboard()
    
    # Generate report
    insights = analyzer.generate_insights_report(report_path)
    
    print()
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS COMPLETE ✅")
    print("="*70)
    
    return insights
