"""
RevenueIQ AI - Interactive Dashboard (Enhanced Version)
Task 3C: Web-based interactive analytics dashboard with advanced filters
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global data holder
data_store = {}


def load_data(data_path='data/processed/transactions_sales_only.csv'):
    """Load transaction data"""
    df = pd.read_csv(data_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    data_store['df'] = df
    return df


def create_kpi_card(title, value, color='#3498db'):
    """Create a KPI card component"""
    return html.Div([
        html.H4(title, style={'color': '#7f8c8d', 'margin': '0'}),
        html.H2(value, style={'color': color, 'margin': '10px 0', 'fontWeight': 'bold'}),
    ], style={
        'padding': '20px',
        'backgroundColor': '#ecf0f1',
        'borderRadius': '10px',
        'textAlign': 'center',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    })


def create_layout(df):
    """Create dashboard layout"""
    
    # Calculate KPIs
    total_revenue = df['TotalPrice'].sum()
    total_transactions = len(df)
    total_customers = df['CustomerID'].nunique()
    avg_order_value = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    
    # Get date range
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    
    # Get unique countries
    countries = ['All'] + sorted(df['Country'].unique().tolist())
    
    # Get top 50 products for dropdown (to keep it manageable)
    top_products = df.groupby('Description')['TotalPrice'].sum().nlargest(50).index.tolist()
    products = ['All'] + sorted(top_products)
    
    layout = html.Div([
        # Header
        html.Div([
            html.H1('RevenueIQ AI - Interactive Dashboard', 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
            html.P('Real-time Business Analytics & Insights (Enhanced Version)',
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
        
        # Filters Section (Enhanced)
        html.Div([
            # Row 1: Date Range and Country
            html.Div([
                html.Div([
                    html.Label('📅 Date Range:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label('🌍 Country:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='country-filter',
                        options=[{'label': c, 'value': c} for c in countries],
                        value='All',
                        placeholder='Select country...',
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'marginBottom': '15px'}),
            
            # Row 2: Product, Customer Type, and Time View
            html.Div([
                html.Div([
                    html.Label('🛍️ Product:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='product-filter',
                        options=[{'label': p[:50], 'value': p} for p in products],
                        value='All',
                        placeholder='Select product...',
                        style={'width': '100%'}
                    )
                ], style={'width': '31%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label('👥 Customer Type:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='customer-type',
                        options=[
                            {'label': 'All Customers', 'value': 'All'},
                            {'label': 'Guest Only', 'value': 'Guest'},
                            {'label': 'Registered Only', 'value': 'Registered'}
                        ],
                        value='All',
                        style={'width': '100%'}
                    )
                ], style={'width': '31%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label('📊 Time View:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='time-view',
                        options=[
                            {'label': ' Daily', 'value': 'Daily'},
                            {'label': ' Weekly', 'value': 'Weekly'},
                            {'label': ' Monthly', 'value': 'Monthly'}
                        ],
                        value='Daily',
                        inline=True,
                        style={'marginTop': '8px'}
                    )
                ], style={'width': '31%', 'display': 'inline-block'}),
            ]),
            
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px', 
                 'borderRadius': '5px', 'border': '1px solid #dee2e6'}),
        
        # KPI Cards Row
        html.Div(id='kpi-cards', children=[
            html.Div([
                create_kpi_card('Total Revenue', f'${total_revenue:,.0f}', '#27ae60'),
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                create_kpi_card('Transactions', f'{total_transactions:,}', '#3498db'),
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                create_kpi_card('Customers', f'{total_customers:,}', '#9b59b6'),
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                create_kpi_card('Avg Order Value', f'${avg_order_value:,.2f}', '#e74c3c'),
            ], style={'width': '23%', 'display': 'inline-block'}),
        ], style={'marginBottom': '30px'}),
        
        # Charts Row 1: Revenue Trend & Top Products
        html.Div([
            html.Div([
                dcc.Graph(id='revenue-trend')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='top-products')
            ], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'marginBottom': '30px'}),
        
        # Charts Row 2: Geographic Distribution & Time Patterns
        html.Div([
            html.Div([
                dcc.Graph(id='geographic-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='hourly-pattern')
            ], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'marginBottom': '30px'}),
        
        # Charts Row 3: Customer Segments & Day of Week
        html.Div([
            html.Div([
                dcc.Graph(id='customer-segments')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='day-of-week')
            ], style={'width': '48%', 'display': 'inline-block'}),
        ]),
        
        # Footer
        html.Div([
            html.P([
                '🔄 Last Updated: ',
                html.Span(id='last-update', children=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            ], style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '40px'}),
            html.P('💡 Tip: Use filters above to drill down into specific segments',
                  style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
        ])
        
    ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'})
    
    return layout


# Enhanced Callbacks for interactivity
@callback(
    [Output('kpi-cards', 'children'),
     Output('revenue-trend', 'figure'),
     Output('top-products', 'figure'),
     Output('geographic-distribution', 'figure'),
     Output('hourly-pattern', 'figure'),
     Output('customer-segments', 'figure'),
     Output('day-of-week', 'figure'),
     Output('last-update', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('country-filter', 'value'),
     Input('product-filter', 'value'),
     Input('customer-type', 'value'),
     Input('time-view', 'value')]
)
def update_dashboard(start_date, end_date, country, product, customer_type, time_view):
    """Update all dashboard components based on filters"""
    
    df = data_store['df'].copy()
    
    # Apply Date Filter
    df = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]
    
    # Apply Country Filter
    if country != 'All':
        df = df[df['Country'] == country]
    
    # Apply Product Filter
    if product != 'All':
        df = df[df['Description'] == product]
    
    # Apply Customer Type Filter
    if customer_type == 'Guest':
        df = df[df['CustomerID'] == 'Guest']
    elif customer_type == 'Registered':
        df = df[df['CustomerID'] != 'Guest']
    
    # Calculate KPIs
    total_revenue = df['TotalPrice'].sum()
    total_transactions = len(df)
    total_customers = df['CustomerID'].nunique()
    avg_order_value = df.groupby('InvoiceNo')['TotalPrice'].sum().mean() if len(df) > 0 else 0
    
    # KPI Cards
    kpi_cards = [
        html.Div([
            create_kpi_card('Total Revenue', f'${total_revenue:,.0f}', '#27ae60'),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            create_kpi_card('Transactions', f'{total_transactions:,}', '#3498db'),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            create_kpi_card('Customers', f'{total_customers:,}', '#9b59b6'),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            create_kpi_card('Avg Order Value', f'${avg_order_value:,.2f}', '#e74c3c'),
        ], style={'width': '23%', 'display': 'inline-block'}),
    ]
    
    # 1. Revenue Trend (with Time View selection)
    if time_view == 'Daily':
        revenue_grouped = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
        revenue_grouped.columns = ['Date', 'Revenue']
        x_label = 'Date'
        title = 'Daily Revenue Trend'
    elif time_view == 'Weekly':
        df['Week'] = df['InvoiceDate'].dt.to_period('W').astype(str)
        revenue_grouped = df.groupby('Week')['TotalPrice'].sum().reset_index()
        revenue_grouped.columns = ['Date', 'Revenue']
        x_label = 'Week'
        title = 'Weekly Revenue Trend'
    else:  # Monthly
        df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        revenue_grouped = df.groupby('Month')['TotalPrice'].sum().reset_index()
        revenue_grouped.columns = ['Date', 'Revenue']
        x_label = 'Month'
        title = 'Monthly Revenue Trend'
    
    fig_revenue = px.line(revenue_grouped, x='Date', y='Revenue',
                         title=title,
                         labels={'Revenue': 'Revenue ($)', 'Date': x_label})
    fig_revenue.update_traces(line_color='#3498db', line_width=2)
    fig_revenue.update_layout(hovermode='x unified')
    
    # 2. Top Products
    top_products = df.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
    top_products.columns = ['Product', 'Revenue']
    
    fig_products = px.bar(top_products, x='Revenue', y='Product',
                         orientation='h',
                         title='Top 10 Products by Revenue',
                         labels={'Revenue': 'Revenue ($)'})
    fig_products.update_traces(marker_color='#27ae60')
    fig_products.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # 3. Geographic Distribution
    country_revenue = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
    country_revenue.columns = ['Country', 'Revenue']
    
    fig_geo = px.pie(country_revenue, values='Revenue', names='Country',
                    title='Revenue by Country (Top 10)',
                    hole=0.4)
    fig_geo.update_traces(textposition='inside', textinfo='percent+label')
    
    # 4. Hourly Pattern
    hourly_revenue = df.groupby('Hour')['TotalPrice'].sum().reset_index()
    hourly_revenue.columns = ['Hour', 'Revenue']
    
    fig_hourly = px.bar(hourly_revenue, x='Hour', y='Revenue',
                       title='Revenue by Hour of Day',
                       labels={'Revenue': 'Revenue ($)', 'Hour': 'Hour of Day'})
    fig_hourly.update_traces(marker_color='#e74c3c')
    fig_hourly.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    
    # 5. Customer Segments
    guest_revenue = df[df['CustomerID'] == 'Guest']['TotalPrice'].sum()
    registered_revenue = df[df['CustomerID'] != 'Guest']['TotalPrice'].sum()
    
    fig_segments = go.Figure(data=[
        go.Bar(name='Guest', x=['Guest'], y=[guest_revenue], marker_color='#f39c12'),
        go.Bar(name='Registered', x=['Registered'], y=[registered_revenue], marker_color='#9b59b6')
    ])
    fig_segments.update_layout(title='Revenue: Guest vs Registered Customers',
                              yaxis_title='Revenue ($)',
                              showlegend=True)
    
    # 6. Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_revenue_dow = df.groupby('DayName')['TotalPrice'].sum().reindex(day_order).reset_index()
    daily_revenue_dow.columns = ['Day', 'Revenue']
    
    fig_dow = px.bar(daily_revenue_dow, x='Day', y='Revenue',
                    title='Revenue by Day of Week',
                    labels={'Revenue': 'Revenue ($)'})
    fig_dow.update_traces(marker_color='#16a085')
    
    # Last update timestamp
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return kpi_cards, fig_revenue, fig_products, fig_geo, fig_hourly, fig_segments, fig_dow, last_update


def run_dashboard(data_path='data/processed/transactions_sales_only.csv', 
                 port=8050, debug=True):
    """
    Run the interactive dashboard
    
    Args:
        data_path: Path to sales data CSV
        port: Port number (default: 8050)
        debug: Enable debug mode
    """
    print("="*70)
    print("REVENUEIQ AI - INTERACTIVE DASHBOARD (ENHANCED)")
    print("="*70)
    print()
    print("📂 Loading data...")
    
    df = load_data(data_path)
    print(f"✅ Loaded {len(df):,} transactions")
    print()
    
    # Set layout
    app.layout = create_layout(df)
    
    print("🚀 Starting enhanced dashboard server...")
    print(f"📊 Dashboard URL: http://127.0.0.1:{port}/")
    print()
    print("="*70)
    print("ENHANCED DASHBOARD FEATURES:")
    print("="*70)
    print("✅ Date Range Picker - Filter by time period")
    print("✅ Country Filter - Focus on specific markets")
    print("✅ Product Filter - Analyze individual products (top 50)")
    print("✅ Customer Type Filter - Guest vs Registered analysis")
    print("✅ Time View Toggle - Daily/Weekly/Monthly aggregation")
    print()
    print("📊 All metrics and charts update automatically!")
    print("🎯 Hover over charts for detailed tooltips")
    print()
    print("Press CTRL+C to stop the server")
    print("="*70)
    print()
    
    # Run server
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_dashboard()
