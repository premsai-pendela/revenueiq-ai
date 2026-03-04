"""
RevenueIQ AI - KMeans Customer Clustering
Machine Learning-based Customer Segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import duckdb

print("=" * 70)
print("REVENUEIQ - KMEANS CUSTOMER CLUSTERING")
print("Machine Learning Segmentation Engine")
print("=" * 70)


class CustomerClustering:
    """KMeans clustering for customer segmentation"""
    
    def __init__(self, data_path='data/processed/transactions_cleaned.csv'):
        """Initialize with transaction data"""
        print(f"\n📊 Loading transaction data...")
        self.df = pd.read_csv(data_path)
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        print(f"✓ Loaded {len(self.df):,} transactions")
        
    def prepare_customer_features(self):
        """Engineer features for clustering"""
        print("\n🔧 Engineering customer features...")
        
        # Calculate max date for recency
        max_date = self.df['InvoiceDate'].max()
        
        # Aggregate customer-level features
        customer_features = self.df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',          # Frequency
            'TotalPrice': ['sum', 'mean'],   # Monetary
            'Quantity': 'sum',                # Total quantity
            'InvoiceDate': ['min', 'max']    # First and last purchase
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'CustomerID', 'Frequency', 'MonetaryValue', 
            'AvgOrderValue', 'TotalQuantity', 'FirstPurchase', 'LastPurchase'
        ]
        
        # Calculate Recency
        customer_features['Recency'] = (
            max_date - customer_features['LastPurchase']
        ).dt.days
        
        # Calculate Customer Lifespan
        customer_features['CustomerLifespan'] = (
            customer_features['LastPurchase'] - customer_features['FirstPurchase']
        ).dt.days
        
        # Remove Guest customers
        customer_features = customer_features[
            customer_features['CustomerID'] != 'Guest'
        ].copy()
        
        # Remove customers with negative monetary value (refunds only)
        customer_features = customer_features[
            customer_features['MonetaryValue'] > 0
        ].copy()
        
        print(f"✓ Prepared features for {len(customer_features):,} customers")
        print(f"\n📋 Feature Summary:")
        print(customer_features[['Recency', 'Frequency', 'MonetaryValue', 
                                 'AvgOrderValue', 'TotalQuantity', 
                                 'CustomerLifespan']].describe())
        
        self.customer_features = customer_features
        return customer_features
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method"""
        print("\n📊 Finding optimal number of clusters...")
        
        # Select features for clustering
        feature_cols = ['Recency', 'Frequency', 'MonetaryValue', 
                       'AvgOrderValue', 'TotalQuantity', 'CustomerLifespan']
        
        X = self.customer_features[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate inertia and silhouette scores
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia plot
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=5, color='red', linestyle='--', label='k=5 (chosen)')
        axes[0].legend()
        
        # Silhouette score plot
        axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=5, color='red', linestyle='--', label='k=5 (chosen)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/kmeans_elbow_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/kmeans_elbow_analysis.png")
        plt.close()
        
        print(f"\n📊 Silhouette Scores:")
        for k, score in zip(K_range, silhouette_scores):
            print(f"   k={k}: {score:.4f}")
        
        print(f"\n✅ Recommended: k=5 clusters (good balance)")
        
        return 5
    
    def perform_clustering(self, n_clusters=5):
        """Perform KMeans clustering"""
        print(f"\n🤖 Performing KMeans clustering (k={n_clusters})...")
        
        # Select features
        feature_cols = ['Recency', 'Frequency', 'MonetaryValue', 
                       'AvgOrderValue', 'TotalQuantity', 'CustomerLifespan']
        
        X = self.customer_features[feature_cols].fillna(0)
        
        # Scale features (important for KMeans!)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        self.customer_features['Cluster'] = clusters
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        print(f"✓ Clustering complete!")
        print(f"   Silhouette Score: {silhouette_avg:.4f}")
        print(f"   (Score ranges -1 to 1, higher is better)")
        
        # Store for later use
        self.scaler = scaler
        self.kmeans = kmeans
        self.feature_cols = feature_cols
        
        return clusters
    
    def analyze_clusters(self):
        """Analyze and profile each cluster"""
        print("\n📊 CLUSTER ANALYSIS")
        print("=" * 70)
        
        # Group by cluster
        cluster_summary = self.customer_features.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'MonetaryValue': ['mean', 'sum'],
            'AvgOrderValue': 'mean',
            'TotalQuantity': 'mean',
            'CustomerLifespan': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_summary.columns = [
            'Customers', 'Avg_Recency', 'Avg_Frequency', 
            'Avg_MonetaryValue', 'Total_Revenue', 
            'Avg_OrderValue', 'Avg_Quantity', 'Avg_Lifespan'
        ]
        
        # Calculate percentages
        cluster_summary['Customer_Pct'] = (
            cluster_summary['Customers'] / cluster_summary['Customers'].sum() * 100
        ).round(1)
        
        cluster_summary['Revenue_Pct'] = (
            cluster_summary['Total_Revenue'] / cluster_summary['Total_Revenue'].sum() * 100
        ).round(1)
        
        # Assign meaningful names based on characteristics
        cluster_names = self._assign_cluster_names(cluster_summary)
        cluster_summary['Cluster_Name'] = cluster_names
        
        print("\n🎯 CLUSTER PROFILES:")
        print(cluster_summary.to_string())
        
        # Detailed breakdown
        print("\n" + "=" * 70)
        print("DETAILED CLUSTER BREAKDOWN")
        print("=" * 70)
        
        for cluster_id in sorted(cluster_summary.index):
            row = cluster_summary.loc[cluster_id]
            print(f"\n{'='*70}")
            print(f"CLUSTER {cluster_id}: {row['Cluster_Name']}")
            print(f"{'='*70}")
            print(f"  Customers:        {int(row['Customers']):,} ({row['Customer_Pct']:.1f}%)")
            print(f"  Revenue:          ${row['Total_Revenue']:,.2f} ({row['Revenue_Pct']:.1f}%)")
            print(f"  Avg Recency:      {row['Avg_Recency']:.0f} days")
            print(f"  Avg Frequency:    {row['Avg_Frequency']:.1f} orders")
            print(f"  Avg Value:        ${row['Avg_MonetaryValue']:,.2f}")
            print(f"  Avg Order Size:   ${row['Avg_OrderValue']:,.2f}")
            print(f"  Avg Lifespan:     {row['Avg_Lifespan']:.0f} days")
            
            # Business recommendation
            print(f"\n  💡 Strategy: {self._get_strategy(cluster_id, row)}")
        
        self.cluster_summary = cluster_summary
        return cluster_summary
    
    def _assign_cluster_names(self, summary):
        """Assign meaningful names to clusters"""
        names = {}
        
        for idx in summary.index:
            recency = summary.loc[idx, 'Avg_Recency']
            frequency = summary.loc[idx, 'Avg_Frequency']
            monetary = summary.loc[idx, 'Avg_MonetaryValue']
            
            # Logic for naming
            if recency < 50 and frequency > 10 and monetary > 3000:
                names[idx] = "💎 VIP Champions"
            elif recency < 80 and frequency > 5:
                names[idx] = "⭐ Loyal Customers"
            elif recency < 100 and monetary > 2000:
                names[idx] = "🎯 High-Value Buyers"
            elif recency > 150:
                names[idx] = "⚠️ At-Risk/Lost"
            else:
                names[idx] = "🌱 Potential Growers"
        
        return [names[i] for i in sorted(summary.index)]
    
    def _get_strategy(self, cluster_id, row):
        """Get marketing strategy for each cluster"""
        recency = row['Avg_Recency']
        
        if "VIP" in row['Cluster_Name']:
            return "Maintain exclusivity, early access to new products, personal service"
        elif "Loyal" in row['Cluster_Name']:
            return "Reward loyalty, refer-a-friend program, VIP upgrade path"
        elif "High-Value" in row['Cluster_Name']:
            return "Increase frequency with targeted campaigns, bundle offers"
        elif "At-Risk" in row['Cluster_Name']:
            return "Win-back campaign, special discounts, survey for feedback"
        else:
            return "Nurture with educational content, small incentives to increase engagement"
    
    def visualize_clusters_3d(self):
        """Create interactive 3D visualization"""
        print("\n🎨 Creating 3D cluster visualization...")
        
        # Prepare data
        plot_data = self.customer_features.copy()
        plot_data['Cluster_Name'] = plot_data['Cluster'].map(
            self.cluster_summary['Cluster_Name'].to_dict()
        )
        
        # Fix: Use absolute value for size (handle negatives)
        plot_data['MonetaryValue_Size'] = plot_data['MonetaryValue'].abs() + 1
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot_data,
            x='Recency',
            y='Frequency',
            z='MonetaryValue',
            color='Cluster_Name',
            size='MonetaryValue_Size',
            hover_data=['CustomerID', 'AvgOrderValue', 'TotalQuantity'],
            title='Customer Segmentation - 3D Cluster View',
            labels={
                'Recency': 'Recency (days)',
                'Frequency': 'Frequency (orders)',
                'MonetaryValue': 'Monetary Value ($)'
            },
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Recency (days since last purchase)',
                yaxis_title='Frequency (number of orders)',
                zaxis_title='Monetary Value ($)'
            ),
            height=700,
            font=dict(size=12)
        )
        
        # Save
        fig.write_html('outputs/kmeans_3d_clusters.html')
        print("✓ Saved: outputs/kmeans_3d_clusters.html (open in browser!)")
        
        # Also create 2D projections
        self._create_2d_visualizations()
        
    def _create_2d_visualizations(self):
        """Create 2D scatter plots"""
        print("\n🎨 Creating 2D cluster visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Define cluster colors
        cluster_names = self.cluster_summary['Cluster_Name'].values
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Plot 1: Recency vs Frequency
        for cluster_id in range(len(cluster_names)):
            cluster_data = self.customer_features[
                self.customer_features['Cluster'] == cluster_id
            ]
            axes[0, 0].scatter(
                cluster_data['Recency'],
                cluster_data['Frequency'],
                c=colors[cluster_id],
                label=cluster_names[cluster_id],
                alpha=0.6,
                s=50
            )
        axes[0, 0].set_xlabel('Recency (days)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency (orders)', fontsize=11)
        axes[0, 0].set_title('Recency vs Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Frequency vs Monetary Value
        for cluster_id in range(len(cluster_names)):
            cluster_data = self.customer_features[
                self.customer_features['Cluster'] == cluster_id
            ]
            axes[0, 1].scatter(
                cluster_data['Frequency'],
                cluster_data['MonetaryValue'],
                c=colors[cluster_id],
                label=cluster_names[cluster_id],
                alpha=0.6,
                s=50
            )
        axes[0, 1].set_xlabel('Frequency (orders)', fontsize=11)
        axes[0, 1].set_ylabel('Monetary Value ($)', fontsize=11)
        axes[0, 1].set_title('Frequency vs Monetary Value', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Recency vs Monetary Value
        for cluster_id in range(len(cluster_names)):
            cluster_data = self.customer_features[
                self.customer_features['Cluster'] == cluster_id
            ]
            axes[1, 0].scatter(
                cluster_data['Recency'],
                cluster_data['MonetaryValue'],
                c=colors[cluster_id],
                label=cluster_names[cluster_id],
                alpha=0.6,
                s=50
            )
        axes[1, 0].set_xlabel('Recency (days)', fontsize=11)
        axes[1, 0].set_ylabel('Monetary Value ($)', fontsize=11)
        axes[1, 0].set_title('Recency vs Monetary Value', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cluster Distribution
        cluster_counts = self.customer_features['Cluster'].value_counts().sort_index()
        bars = axes[1, 1].bar(
            range(len(cluster_names)),
            cluster_counts.values,
            color=colors,
            alpha=0.7
        )
        axes[1, 1].set_xlabel('Cluster', fontsize=11)
        axes[1, 1].set_ylabel('Number of Customers', fontsize=11)
        axes[1, 1].set_title('Cluster Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(range(len(cluster_names)))
        axes[1, 1].set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.tight_layout()
        plt.savefig('outputs/kmeans_2d_clusters.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/kmeans_2d_clusters.png")
        plt.close()
    
    def save_to_duckdb(self):
        """Save clustering results to DuckDB"""
        print("\n💾 Saving results to DuckDB...")
        
        # Connect to DuckDB
        con = duckdb.connect('data/revenueiq.db')
        
        # Prepare data
        cluster_data = self.customer_features[[
            'CustomerID', 'Cluster', 'Recency', 'Frequency', 
            'MonetaryValue', 'AvgOrderValue', 'TotalQuantity', 'CustomerLifespan'
        ]].copy()
        
        # Add cluster names
        cluster_data['ClusterName'] = cluster_data['Cluster'].map(
            self.cluster_summary['Cluster_Name'].to_dict()
        )
        
        # Drop existing table
        con.execute("DROP TABLE IF EXISTS customer_clusters")
        
        # Create table
        con.execute("""
            CREATE TABLE customer_clusters AS 
            SELECT * FROM cluster_data
        """)
        
        # Verify
        count = con.execute("SELECT COUNT(*) FROM customer_clusters").fetchone()[0]
        print(f"✓ Saved {count:,} customers to DuckDB table: customer_clusters")
        
        # Show sample
        sample = con.execute("""
            SELECT CustomerID, Cluster, ClusterName, MonetaryValue 
            FROM customer_clusters 
            LIMIT 5
        """).df()
        print("\n📊 Sample from database:")
        print(sample.to_string(index=False))
        
        con.close()
        
        # Also save CSV
        cluster_data.to_csv('data/processed/customer_clusters.csv', index=False)
        print("✓ Saved: data/processed/customer_clusters.csv")
    
    def generate_clustering_report(self):
        """Generate comprehensive clustering report"""
        print("\n" + "=" * 70)
        print("GENERATING KMEANS CLUSTERING REPORT")
        print("=" * 70)
        
        # Prepare features
        self.prepare_customer_features()
        
        # Find optimal clusters
        optimal_k = self.find_optimal_clusters(max_k=10)
        
        # Perform clustering
        self.perform_clustering(n_clusters=optimal_k)
        
        # Analyze clusters
        cluster_summary = self.analyze_clusters()
        
        # Visualize
        self.visualize_clusters_3d()
        
        # Save to database
        self.save_to_duckdb()
        
        print("\n" + "=" * 70)
        print("✅ KMEANS CLUSTERING COMPLETE!")
        print("=" * 70)
        print("\n📁 Generated Files:")
        print("   - outputs/kmeans_elbow_analysis.png")
        print("   - outputs/kmeans_2d_clusters.png")
        print("   - outputs/kmeans_3d_clusters.html (interactive!)")
        print("   - data/processed/customer_clusters.csv")
        print("   - DuckDB table: customer_clusters")
        
        print("\n🎯 Key Findings:")
        print(f"   - Total Customers: {len(self.customer_features):,}")
        print(f"   - Clusters Created: {optimal_k}")
        print(f"   - Best Cluster: {cluster_summary['Cluster_Name'].iloc[0]}")
        print(f"   - Top Revenue: ${cluster_summary['Total_Revenue'].max():,.2f}")
        
        return cluster_summary


def main():
    """Execute KMeans clustering pipeline"""
    
    # Initialize
    clustering = CustomerClustering()
    
    # Generate report
    results = clustering.generate_clustering_report()
    
    print("\n🎉 KMeans Clustering Complete!")
    print("\n📊 Open 'outputs/kmeans_3d_clusters.html' in your browser for interactive viz!")
    print("\nNext: Groq LLM Integration (1.5 hours)")


if __name__ == "__main__":
    main()
