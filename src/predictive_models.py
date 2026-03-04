"""
RevenueIQ AI - Predictive Analytics Module
Task 4: Machine Learning Models for Sales & Customer Insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

print("=" * 70)
print("REVENUEIQ AI - PREDICTIVE ANALYTICS ENGINE")
print("Task 4: Machine Learning Models")
print("=" * 70)

class RevenuePredictor:
    """Advanced predictive analytics for revenue intelligence"""
    
    def __init__(self, data_path):
        """Initialize with cleaned transaction data"""
        print("\n📊 Loading data for predictive modeling...")
        self.df = pd.read_csv(data_path)
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        print(f"✓ Loaded {len(self.df):,} transactions")
        
    def prepare_time_series_data(self):
        """Prepare daily sales data for forecasting"""
        print("\n🔄 Preparing time series data...")
        
        # Daily aggregation
        daily_sales = self.df.groupby(self.df['InvoiceDate'].dt.date).agg({
            'TotalPrice': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).reset_index()
        
        daily_sales.columns = ['Date', 'Revenue', 'Orders', 'Customers']
        daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
        daily_sales = daily_sales.set_index('Date').sort_index()
        
        print(f"✓ Created daily time series: {len(daily_sales)} days")
        return daily_sales
    
    def forecast_sales_arima(self, days_ahead=30):
        """ARIMA model for sales forecasting"""
        print("\n📈 MODEL 1: ARIMA Sales Forecasting")
        print("-" * 50)
        
        daily_sales = self.prepare_time_series_data()
        
        # Train/Test split
        train_size = int(len(daily_sales) * 0.8)
        train, test = daily_sales['Revenue'][:train_size], daily_sales['Revenue'][train_size:]
        
        print(f"Training set: {len(train)} days")
        print(f"Test set: {len(test)} days")
        
        # Fit ARIMA model
        print("\n🔧 Fitting ARIMA(1,1,1) model...")
        model = ARIMA(train, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(test))
        
        # Metrics
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        
        print(f"\n📊 Model Performance:")
        print(f"   MAE:  ${mae:,.2f}")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Future forecast
        future_forecast = fitted_model.forecast(steps=days_ahead)
        
        # Visualization
        self._plot_forecast(train, test, forecast, future_forecast, 'ARIMA Sales Forecast')
        
        return {
            'model': fitted_model,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'future_forecast': future_forecast
        }
    
    def forecast_sales_exponential(self, days_ahead=30):
        """Exponential Smoothing for sales forecasting"""
        print("\n📈 MODEL 2: Exponential Smoothing Forecast")
        print("-" * 50)
        
        daily_sales = self.prepare_time_series_data()
        
        # Train/Test split
        train_size = int(len(daily_sales) * 0.8)
        train, test = daily_sales['Revenue'][:train_size], daily_sales['Revenue'][train_size:]
        
        # Fit model with trend and seasonality
        print("🔧 Fitting Holt-Winters model...")
        model = ExponentialSmoothing(
            train, 
            seasonal_periods=7,  # Weekly seasonality
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(test))
        
        # Metrics
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        
        print(f"\n📊 Model Performance:")
        print(f"   MAE:  ${mae:,.2f}")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Future forecast
        future_forecast = fitted_model.forecast(steps=days_ahead)
        
        print(f"\n🔮 30-Day Revenue Forecast:")
        print(f"   Total Predicted: ${future_forecast.sum():,.2f}")
        print(f"   Daily Average: ${future_forecast.mean():,.2f}")
        
        return {
            'model': fitted_model,
            'forecast': future_forecast,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def predict_customer_churn(self):
        """Random Forest model for churn prediction"""
        print("\n🎯 MODEL 3: Customer Churn Prediction")
        print("-" * 50)
        
        # Prepare customer features
        print("🔄 Engineering customer features...")
        
        customer_features = self.df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',  # Purchase frequency
            'TotalPrice': ['sum', 'mean'],  # Monetary value
            'Quantity': 'sum',
            'InvoiceDate': ['min', 'max']
        }).reset_index()
        
        customer_features.columns = ['CustomerID', 'PurchaseFrequency', 
                                     'TotalSpent', 'AvgOrderValue', 
                                     'TotalQuantity', 'FirstPurchase', 'LastPurchase']
        
        # Calculate recency
        max_date = self.df['InvoiceDate'].max()
        customer_features['Recency'] = (max_date - customer_features['LastPurchase']).dt.days
        customer_features['CustomerLifetime'] = (customer_features['LastPurchase'] - 
                                                 customer_features['FirstPurchase']).dt.days
        
        # Define churn (no purchase in last 90 days)
        customer_features['Churned'] = (customer_features['Recency'] > 90).astype(int)
        
        print(f"\n📊 Churn Distribution:")
        print(f"   Active: {(customer_features['Churned']==0).sum():,} customers")
        print(f"   Churned: {(customer_features['Churned']==1).sum():,} customers")
        print(f"   Churn Rate: {customer_features['Churned'].mean()*100:.1f}%")
        
        # Prepare features
        feature_cols = ['PurchaseFrequency', 'TotalSpent', 'AvgOrderValue', 
                       'TotalQuantity', 'Recency', 'CustomerLifetime']
        
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['Churned']
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🎯 Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"   {row['Feature']:<20} {row['Importance']:.4f}")
        
        # Identify at-risk customers
        customer_features['ChurnProbability'] = rf_model.predict_proba(
            scaler.transform(customer_features[feature_cols].fillna(0))
        )[:, 1]
        
        at_risk = customer_features[
            (customer_features['Churned'] == 0) & 
            (customer_features['ChurnProbability'] > 0.7)
        ].sort_values('ChurnProbability', ascending=False)
        
        print(f"\n⚠️  At-Risk Customers: {len(at_risk)}")
        print(f"   Potential Revenue Loss: ${at_risk['TotalSpent'].sum():,.2f}")
        
        # Visualization
        self._plot_churn_analysis(customer_features, feature_importance)
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'at_risk_customers': at_risk,
            'feature_importance': feature_importance
        }
    
    def forecast_product_demand(self, top_n=20):
        """Predict demand for top products"""
        print("\n📦 MODEL 4: Product Demand Forecasting")
        print("-" * 50)
        
        # Get top products
        top_products = self.df.groupby('Description')['Quantity'].sum().nlargest(top_n)
        
        print(f"🔧 Forecasting demand for top {top_n} products...")
        
        forecasts = []
        
        for product in top_products.index[:5]:  # Show top 5
            # Product time series
            product_data = self.df[self.df['Description'] == product].copy()
            daily_demand = product_data.groupby(
                product_data['InvoiceDate'].dt.date
            )['Quantity'].sum()
            
            # Simple moving average forecast
            ma_7 = daily_demand.rolling(window=7).mean()
            forecast_value = ma_7.iloc[-1] if not ma_7.empty else daily_demand.mean()
            
            forecasts.append({
                'Product': product,
                'Avg_Daily_Demand': daily_demand.mean(),
                'Forecasted_Demand': forecast_value,
                'Total_Sold': top_products[product]
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        print("\n📊 Top Product Demand Forecasts:")
        print(forecast_df.to_string(index=False))
        
        return forecast_df
    
    def detect_anomalies(self):
        """Detect anomalous transactions using Isolation Forest"""
        print("\n🚨 MODEL 5: Anomaly Detection")
        print("-" * 50)
        
        # Prepare features
        print("🔄 Preparing transaction features...")
        
        anomaly_features = self.df[['Quantity', 'UnitPrice', 'TotalPrice']].copy()
        
        # Add time-based features
        anomaly_features['Hour'] = self.df['InvoiceDate'].dt.hour
        anomaly_features['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        
        # Remove nulls
        anomaly_features = anomaly_features.dropna()
        
        # Train Isolation Forest
        print("🌲 Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.01,  # Expect 1% anomalies
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(anomaly_features)
        
        # Identify anomalies
        anomaly_features['Anomaly'] = predictions
        anomalies = anomaly_features[anomaly_features['Anomaly'] == -1]
        
        print(f"\n📊 Anomaly Detection Results:")
        print(f"   Total Transactions: {len(anomaly_features):,}")
        print(f"   Anomalies Detected: {len(anomalies):,}")
        print(f"   Anomaly Rate: {len(anomalies)/len(anomaly_features)*100:.2f}%")
        
        print(f"\n🚨 Anomaly Characteristics:")
        print(f"   Avg Quantity: {anomalies['Quantity'].mean():.1f} (Normal: {anomaly_features[anomaly_features['Anomaly']==1]['Quantity'].mean():.1f})")
        print(f"   Avg Price: ${anomalies['TotalPrice'].mean():.2f} (Normal: ${anomaly_features[anomaly_features['Anomaly']==1]['TotalPrice'].mean():.2f})")
        
        # Visualization
        self._plot_anomalies(anomaly_features)
        
        return {
            'model': iso_forest,
            'anomalies': anomalies,
            'anomaly_rate': len(anomalies)/len(anomaly_features)
        }
    
    def _plot_forecast(self, train, test, forecast, future_forecast, title):
        """Plot forecast results"""
        plt.figure(figsize=(14, 6))
        
        plt.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test.index, test.values, label='Actual', color='green', linewidth=2)
        plt.plot(test.index, forecast.values, label='Forecast', color='red', 
                linestyle='--', linewidth=2)
        
        # Future forecast
        future_dates = pd.date_range(
            start=test.index[-1] + timedelta(days=1), 
            periods=len(future_forecast)
        )
        plt.plot(future_dates, future_forecast.values, label='Future Forecast', 
                color='orange', linestyle='--', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Revenue ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/sales_forecast.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/sales_forecast.png")
        plt.close()
    
    def _plot_churn_analysis(self, customer_features, feature_importance):
        """Plot churn analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Churn distribution
        churn_counts = customer_features['Churned'].value_counts()
        axes[0].bar(['Active', 'Churned'], churn_counts.values, 
                   color=['green', 'red'], alpha=0.7)
        axes[0].set_title('Customer Churn Distribution', fontweight='bold')
        axes[0].set_ylabel('Number of Customers')
        
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Feature importance
        axes[1].barh(feature_importance['Feature'], 
                    feature_importance['Importance'], 
                    color='steelblue', alpha=0.7)
        axes[1].set_title('Churn Prediction - Feature Importance', fontweight='bold')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('outputs/churn_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/churn_analysis.png")
        plt.close()
    
    def _plot_anomalies(self, anomaly_features):
        """Plot anomaly detection"""
        plt.figure(figsize=(12, 6))
        
        normal = anomaly_features[anomaly_features['Anomaly'] == 1]
        anomalies = anomaly_features[anomaly_features['Anomaly'] == -1]
        
        plt.scatter(normal['Quantity'], normal['TotalPrice'], 
                   c='blue', alpha=0.3, s=10, label='Normal')
        plt.scatter(anomalies['Quantity'], anomalies['TotalPrice'], 
                   c='red', alpha=0.7, s=30, label='Anomaly')
        
        plt.xlabel('Quantity')
        plt.ylabel('Total Price ($)')
        plt.title('Anomaly Detection: Transaction Analysis', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/anomaly_detection.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/anomaly_detection.png")
        plt.close()
    
    def generate_ml_report(self):
        """Generate comprehensive ML report"""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE ML REPORT")
        print("=" * 70)
        
        # Run all models
        arima_results = self.forecast_sales_arima(days_ahead=30)
        exp_results = self.forecast_sales_exponential(days_ahead=30)
        churn_results = self.predict_customer_churn()
        demand_results = self.forecast_product_demand()
        anomaly_results = self.detect_anomalies()
        
        # Save results
        churn_results['at_risk_customers'].to_csv(
            'data/processed/at_risk_customers.csv', index=False
        )
        demand_results.to_csv(
            'data/processed/demand_forecast.csv', index=False
        )
        
        print("\n" + "=" * 70)
        print("✅ PREDICTIVE ANALYTICS COMPLETE!")
        print("=" * 70)
        print("\n📁 Saved Files:")
        print("   - data/processed/at_risk_customers.csv")
        print("   - data/processed/demand_forecast.csv")
        print("   - outputs/sales_forecast.png")
        print("   - outputs/churn_analysis.png")
        print("   - outputs/anomaly_detection.png")
        
        return {
            'sales_forecast': exp_results,
            'churn_prediction': churn_results,
            'demand_forecast': demand_results,
            'anomaly_detection': anomaly_results
        }


def main():
    """Execute predictive analytics pipeline"""
    
    # Initialize
    predictor = RevenuePredictor('data/processed/transactions_sales_only.csv')
    
    # Run all models
    results = predictor.generate_ml_report()
    
    print("\n🎉 Task 4 Complete! All ML models trained and evaluated.")
    print("\nNext: Task 5 - Build AI recommendation engine!")


if __name__ == "__main__":
    main()
