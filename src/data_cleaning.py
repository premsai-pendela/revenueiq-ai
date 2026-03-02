"""
RevenueIQ AI - Data Cleaning Module
Task 2: Handles data quality issues and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and preprocessing for transaction data"""
    
    def __init__(self, df):
        """
        Initialize DataCleaner
        
        Args:
            df: pandas DataFrame with transaction data
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_stats = {}
        
    def handle_missing_customers(self, strategy='guest'):
        """
        Handle missing CustomerIDs
        
        Args:
            strategy: 'guest' (default) - mark as 'Guest', 
                     'remove' - drop these rows
        """
        logger.info("Handling missing CustomerIDs...")
        
        missing_count = self.df['CustomerID'].isna().sum()
        self.cleaning_stats['missing_customers_original'] = missing_count
        
        if strategy == 'guest':
            self.df['CustomerID'] = self.df['CustomerID'].fillna('Guest')
            logger.info(f"Marked {missing_count:,} transactions as 'Guest'")
        elif strategy == 'remove':
            self.df = self.df.dropna(subset=['CustomerID'])
            logger.info(f"Removed {missing_count:,} transactions without CustomerID")
        
        self.cleaning_stats['missing_customers_strategy'] = strategy
        
        return self
    
    def separate_returns(self):
        """Separate returns/cancellations from normal sales"""
        logger.info("Separating returns and cancellations...")
        
        # Identify returns (negative quantity)
        self.df['IsReturn'] = self.df['Quantity'] < 0
        
        returns_count = self.df['IsReturn'].sum()
        self.cleaning_stats['returns_identified'] = returns_count
        
        logger.info(f"Identified {returns_count:,} return transactions")
        
        return self
    
    def remove_bad_prices(self, min_price=0.01, max_price=10000):
        """
        Remove transactions with invalid prices
        
        Args:
            min_price: Minimum valid price (default: 0.01)
            max_price: Maximum valid price (default: 10000)
        """
        logger.info("Removing invalid prices...")
        
        initial_count = len(self.df)
        
        # Remove zero or negative prices
        bad_price_mask = (self.df['UnitPrice'] <= 0) | (self.df['UnitPrice'] > max_price)
        bad_prices_count = bad_price_mask.sum()
        
        self.df = self.df[~bad_price_mask]
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['bad_prices_removed'] = removed_count
        
        logger.info(f"Removed {removed_count:,} transactions with invalid prices")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate transactions"""
        logger.info("Removing duplicate transactions...")
        
        initial_count = len(self.df)
        
        # Define duplicate criteria
        duplicate_cols = ['InvoiceNo', 'StockCode', 'CustomerID', 
                         'Quantity', 'UnitPrice', 'InvoiceDate']
        
        self.df = self.df.drop_duplicates(subset=duplicate_cols, keep='first')
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['duplicates_removed'] = removed_count
        
        logger.info(f"Removed {removed_count:,} duplicate transactions")
        
        return self
    
    def handle_missing_descriptions(self, strategy='unknown'):
        """
        Handle missing product descriptions
        
        Args:
            strategy: 'unknown' - fill with 'Unknown Product',
                     'remove' - drop these rows
        """
        logger.info("Handling missing descriptions...")
        
        missing_count = self.df['Description'].isna().sum()
        self.cleaning_stats['missing_descriptions_original'] = missing_count
        
        if strategy == 'unknown':
            self.df['Description'] = self.df['Description'].fillna('Unknown Product')
            logger.info(f"Filled {missing_count:,} missing descriptions")
        elif strategy == 'remove':
            self.df = self.df.dropna(subset=['Description'])
            logger.info(f"Removed {missing_count:,} transactions without description")
        
        self.cleaning_stats['missing_descriptions_strategy'] = strategy
        
        return self
    
    def create_calculated_fields(self):
        """Create new calculated fields"""
        logger.info("Creating calculated fields...")
        
        # Total price per line item
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        
        # Extract date components
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['Day'] = self.df['InvoiceDate'].dt.day
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        
        # Create month name for readability
        self.df['MonthName'] = self.df['InvoiceDate'].dt.strftime('%B')
        self.df['DayName'] = self.df['InvoiceDate'].dt.strftime('%A')
        
        # Create Year-Month for aggregations
        self.df['YearMonth'] = self.df['InvoiceDate'].dt.to_period('M').astype(str)
        
        logger.info("Created calculated fields: TotalPrice, date components")
        
        return self
    
    def clean_text_fields(self):
        """Clean and standardize text fields"""
        logger.info("Cleaning text fields...")
        
        # Clean Description - remove extra whitespace, convert to title case
        self.df['Description'] = (self.df['Description']
                                  .str.strip()
                                  .str.replace(r'\s+', ' ', regex=True))
        
        # Clean Country names
        if 'Country' in self.df.columns:
            self.df['Country'] = self.df['Country'].str.strip()
        
        logger.info("Cleaned text fields")
        
        return self
    
    def validate_data(self):
        """Perform final validation checks"""
        logger.info("Validating cleaned data...")
        
        validations = {
            'no_null_customerid': self.df['CustomerID'].notna().all(),
            'no_null_description': self.df['Description'].notna().all(),
            'positive_prices': (self.df['UnitPrice'] > 0).all(),
            'valid_quantities': self.df['Quantity'].notna().all(),
            'has_total_price': 'TotalPrice' in self.df.columns,
            'has_date_fields': all(col in self.df.columns for col in 
                                  ['Year', 'Month', 'Day', 'YearMonth'])
        }
        
        self.cleaning_stats['validations'] = validations
        
        all_passed = all(validations.values())
        
        if all_passed:
            logger.info("✅ All validation checks passed!")
        else:
            failed = [k for k, v in validations.items() if not v]
            logger.warning(f"⚠️ Failed validations: {failed}")
        
        return self
    
    def get_cleaning_summary(self):
        """Generate summary of cleaning operations"""
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': len(self.df),
            'final_columns': len(self.df.columns),
            'rows_removed': self.original_shape[0] - len(self.df),
            'removal_percentage': ((self.original_shape[0] - len(self.df)) / 
                                  self.original_shape[0] * 100),
            **self.cleaning_stats
        }
        
        return summary
    
    def get_cleaned_data(self):
        """Return cleaned DataFrame"""
        return self.df


def clean_transaction_data(df, remove_returns=False):
    """
    Main cleaning pipeline
    
    Args:
        df: Raw transaction DataFrame
        remove_returns: If True, remove return transactions (default: False)
    
    Returns:
        Cleaned DataFrame and cleaning summary
    """
    logger.info("="*60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("="*60)
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Run cleaning steps
    (cleaner
     .handle_missing_customers(strategy='guest')
     .separate_returns()
     .remove_bad_prices(min_price=0.01, max_price=10000)
     .remove_duplicates()
     .handle_missing_descriptions(strategy='unknown')
     .create_calculated_fields()
     .clean_text_fields()
     .validate_data())
    
    # Optionally remove returns
    if remove_returns:
        initial_count = len(cleaner.df)
        cleaner.df = cleaner.df[~cleaner.df['IsReturn']]
        removed = initial_count - len(cleaner.df)
        logger.info(f"Removed {removed:,} return transactions")
        cleaner.cleaning_stats['returns_removed'] = removed
    
    # Get results
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_cleaning_summary()
    
    logger.info("="*60)
    logger.info("CLEANING PIPELINE COMPLETED")
    logger.info("="*60)
    
    return cleaned_df, summary


def generate_cleaning_report(summary, output_path):
    """
    Generate detailed cleaning report
    
    Args:
        summary: Cleaning summary dictionary
        output_path: Path to save report
    """
    report_lines = [
        "="*70,
        "REVENUEIQ AI - DATA CLEANING REPORT",
        "TASK 2: Data Cleaning & Preprocessing",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
        "",
        "DATASET OVERVIEW",
        "-"*70,
        f"Original Rows:      {summary['original_rows']:,}",
        f"Original Columns:   {summary['original_columns']:,}",
        f"Final Rows:         {summary['final_rows']:,}",
        f"Final Columns:      {summary['final_columns']:,}",
        f"Rows Removed:       {summary['rows_removed']:,}",
        f"Removal Rate:       {summary['removal_percentage']:.2f}%",
        "",
        "CLEANING OPERATIONS",
        "-"*70,
    ]
    
    # Missing CustomerIDs
    if 'missing_customers_original' in summary:
        report_lines.extend([
            f"Missing CustomerIDs:",
            f"  - Original count: {summary['missing_customers_original']:,}",
            f"  - Strategy: {summary['missing_customers_strategy']}",
            ""
        ])
    
    # Returns
    if 'returns_identified' in summary:
        report_lines.extend([
            f"Returns/Cancellations:",
            f"  - Identified: {summary['returns_identified']:,}",
        ])
        if 'returns_removed' in summary:
            report_lines.append(f"  - Removed: {summary['returns_removed']:,}")
        report_lines.append("")
    
    # Bad prices
    if 'bad_prices_removed' in summary:
        report_lines.extend([
            f"Invalid Prices:",
            f"  - Removed: {summary['bad_prices_removed']:,}",
            ""
        ])
    
    # Duplicates
    if 'duplicates_removed' in summary:
        report_lines.extend([
            f"Duplicate Transactions:",
            f"  - Removed: {summary['duplicates_removed']:,}",
            ""
        ])
    
    # Missing descriptions
    if 'missing_descriptions_original' in summary:
        report_lines.extend([
            f"Missing Descriptions:",
            f"  - Original count: {summary['missing_descriptions_original']:,}",
            f"  - Strategy: {summary['missing_descriptions_strategy']}",
            ""
        ])
    
    # Validations
    if 'validations' in summary:
        report_lines.extend([
            "DATA VALIDATION",
            "-"*70,
        ])
        for check, passed in summary['validations'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report_lines.append(f"{status} - {check}")
        report_lines.append("")
    
    report_lines.extend([
        "="*70,
        "CLEANING COMPLETED SUCCESSFULLY",
        "="*70
    ])
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Cleaning report saved to: {output_path}")
    
    # Also print to console
    print('\n'.join(report_lines))
