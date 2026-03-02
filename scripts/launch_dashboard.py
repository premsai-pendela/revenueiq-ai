"""
RevenueIQ AI - Dashboard Launcher
Task 3C: Launch interactive web dashboard
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dashboard import run_dashboard

# Paths
DATA_PATH = project_root / 'data' / 'processed' / 'transactions_sales_only.csv'

# Launch dashboard
if __name__ == '__main__':
    run_dashboard(
        data_path=str(DATA_PATH),
        port=8050,
        debug=True
    )
