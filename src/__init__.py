"""
Stock Trading Optimization with Dynamic Programming

A comprehensive Python implementation for optimizing stock trading profits
using Dynamic Programming algorithms with k-transaction constraints.

Based on the research paper:
"Dynamic Programming Approach for Optimizing Stock Trading Profits with K-Transactions"
by Danendra Shafi Athallah, Institut Teknologi Bandung

Features:
- Dynamic Programming implementation for k-transaction optimization
- Real market data integration via Alpha Vantage API
- Comprehensive backtesting framework
- Risk analysis and performance metrics
- Strategy comparison and parameter optimization
- Interactive visualization and reporting

Author: Danendra Shafi Athallah
Email: danendra1967@gmail.com, 13523136@std.stei.itb.ac.id
Institution: Institut Teknologi Bandung
"""

__version__ = "1.0.0"
__author__ = "Danendra Shafi Athallah"
__email__ = "danendra1967@gmail.com"
__institution__ = "Institut Teknologi Bandung"
__license__ = "MIT"

# Core algorithms
from .models.dynamic_programing import DynamicProgrammingTrader, DPPortfolioOptimizer
from .models.baseline_strategies import (
    BuyAndHoldStrategy, 
    MovingAverageCrossoverStrategy, 
    MomentumStrategy,
    StrategyComparator
)

# Data handling
from .data.api_client import AlphaVantageClient, DataValidator
from .data.data_downloader import StockDataDownloader
from .data.data_preprocessor import StockDataPreprocessor, FeatureEngineer
from .data.technical_indicators import TechnicalIndicators, TechnicalIndicatorSuite

# Analysis and optimization
from .optimization.backtesting import BacktestEngine, BacktestConfig
from .optimization.parameter_tuning import DPParameterOptimizer, OptimizationConfig
from .analysis.performance_metrics import PerformanceAnalyzer, RiskAnalyzer

# Utilities
from .utils.logger import setup_global_logging, get_logger, TradingLogger
from .utils.helpers import load_config, save_json, ensure_directory

# Version info
def get_version_info():
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'institution': __institution__,
        'license': __license__
    }

# Quick start example
def quick_start_example():
    """
    Print a quick start example for users.
    """
    example_code = """
# Quick Start Example for Stock Trading Optimization

from src import DynamicProgrammingTrader, AlphaVantageClient

# 1. Initialize API client
client = AlphaVantageClient("YOUR_API_KEY")

# 2. Download stock data
stock_data = client.get_stock_data("AAPL")
prices = stock_data['adjusted_close'].tolist()

# 3. Create DP trader and optimize
trader = DynamicProgrammingTrader(max_transactions=5)
max_profit, trades = trader.optimize_profit(prices)

# 4. View results
print(f"Maximum Profit: ${max_profit:.2f}")
print(f"Number of Trades: {len(trades)}")

# 5. Run comprehensive backtesting
backtest_results = trader.backtest(prices)
print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}")
"""
    print(example_code)

# Expose main classes at package level
__all__ = [
    # Core algorithms
    'DynamicProgrammingTrader',
    'DPPortfolioOptimizer',
    'BuyAndHoldStrategy',
    'MovingAverageCrossoverStrategy',
    'MomentumStrategy',
    'StrategyComparator',
    
    # Data handling
    'AlphaVantageClient',
    'DataValidator',
    'StockDataDownloader',
    'StockDataPreprocessor',
    'FeatureEngineer',
    'TechnicalIndicators',
    'TechnicalIndicatorSuite',
    
    # Analysis and optimization
    'BacktestEngine',
    'BacktestConfig',
    'DPParameterOptimizer',
    'OptimizationConfig',
    'PerformanceAnalyzer',
    'RiskAnalyzer',
    
    # Utilities
    'setup_global_logging',
    'get_logger',
    'TradingLogger',
    'load_config',
    'save_json',
    'ensure_directory',
    
    # Meta functions
    'get_version_info',
    'quick_start_example'
]


# Package-level configuration
DEFAULT_CONFIG = {
    'api': {
        'alpha_vantage': {
            'rate_limit': 25,
            'outputsize': 'full',
            'datatype': 'json'
        }
    },
    'strategies': {
        'dp': {
            'max_transactions': [2, 5, 10],
            'transaction_cost': 0.001
        }
    },
    'backtesting': {
        'initial_capital': 100000,
        'train_ratio': 0.7,
        'validation_ratio': 0.15,
        'test_ratio': 0.15
    },
    'logging': {
        'level': 'INFO',
        'enable_console': True,
        'enable_file': True
    }
}


def validate_environment():
    """
    Validate the environment and dependencies.
    
    Returns:
        dict: Validation results
    """
    import sys
    import importlib
    
    validation_results = {
        'python_version': sys.version,
        'python_version_ok': sys.version_info >= (3, 8),
        'required_packages': {},
        'optional_packages': {},
        'environment_ok': True
    }
    
    # Required packages
    required_packages = [
        'pandas', 'numpy', 'requests', 'pyyaml', 
        'matplotlib', 'seaborn', 'statsmodels', 'scipy'
    ]
    
    for package in required_packages:
        try:
            mod = importlib.import_module(package)
            validation_results['required_packages'][package] = {
                'available': True,
                'version': getattr(mod, '__version__', 'unknown')
            }
        except ImportError:
            validation_results['required_packages'][package] = {
                'available': False,
                'version': None
            }
            validation_results['environment_ok'] = False
    
    # Optional packages
    optional_packages = ['plotly', 'jupyter', 'pytest']
    
    for package in optional_packages:
        try:
            mod = importlib.import_module(package)
            validation_results['optional_packages'][package] = {
                'available': True,
                'version': getattr(mod, '__version__', 'unknown')
            }
        except ImportError:
            validation_results['optional_packages'][package] = {
                'available': False,
                'version': None
            }
    
    return validation_results


# Module initialization
def _initialize_package():
    """Initialize package-level settings."""
    import logging
    import warnings
    
    # Filter warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Run initialization
_initialize_package()

# Print package info when imported
print(f"""
Stock Trading Optimization v{__version__}
Dynamic Programming Approach for K-Transaction Profit Optimization

Author: {__author__}
Institution: {__institution__}

Use quick_start_example() to see usage examples.
Use validate_environment() to check dependencies.
""")