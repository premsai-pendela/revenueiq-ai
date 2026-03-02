"""
Data loading utilities for RevenueIQ AI
Optimized for MacBook M4 Air
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

class DataLoader:
    """Handle data loading and basic operations"""
    
    def __init__(self, data_path='data/raw/Online Retail.xlsx'):
        """
        Initialize DataLoader
        
        Args:
            data_path (str): Path to the Excel file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self, verbose=True):
        """
        Load the Online Retail dataset
        
        Args:
            verbose (bool): Print loading information
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if verbose:
            print(f"📂 Loading data from {self.data_path}...")
        
        try:
            # Check if file exists
            if not self.data_path.exists():
                print(f"❌ Error: File not found at {self.data_path}")
                print(f"   Current directory: {Path.cwd()}")
                print(f"   Looking for: {self.data_path.absolute()}")
                return None
            
            # Load Excel file
            self.df = pd.read_excel(
                self.data_path,
                engine='openpyxl',
                parse_dates=['InvoiceDate']
            )
            
            if verbose:
                print(f"✅ Data loaded successfully!")
                print(f"   Rows: {self.df.shape[0]:,}")
                print(f"   Columns: {self.df.shape[1]}")
                memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
                print(f"   Memory: {memory_mb:.2f} MB")
            
            return self.df
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def get_basic_info(self):
        """Get basic information about the dataset"""
        if self.df is None:
            print("⚠️  No data loaded. Call load_data() first.")
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': self.df['InvoiceDate'].min(),
                'end': self.df['InvoiceDate'].max()
            }
        }
        return info
    
    def preview(self, n=5):
        """Preview first n rows"""
        if self.df is None:
            print("⚠️  No data loaded.")
            return None
        return self.df.head(n)
