"""
Simplified Data Download Manager for Stock Trading Optimization.
Updated to work with Alpha Vantage free tier endpoints.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


class StockDataDownloader:
    """
    Data download and management system compatible with Alpha Vantage free tier.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize data downloader with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Setup directories
        self._setup_directories()
        
        # Initialize API client
        from .api_client import AlphaVantageClient, DataValidator
        
        self.api_client = AlphaVantageClient(
            api_key=self.config['api']['alpha_vantage']['api_key'],
            cache_dir=self.config['data']['cache_dir'],
            rate_limit=self.config['api']['alpha_vantage']['rate_limit']
        )
        
        self.data_validator = DataValidator()
        
        # Log free tier usage info
        logger.info("Using Alpha Vantage free tier - TIME_SERIES_DAILY endpoint")
        logger.info("Note: Using close price as proxy for adjusted_close price")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.config['data']['cache_dir'],
            self.config['data']['processed_dir'],
            "data/raw/daily_prices",
            "data/processed/features",
            "data/processed/model_inputs",
            "data/processed/technical_indicators"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def download_single_stock(self, symbol: str, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Download and process data for a single stock.
        
        Args:
            symbol (str): Stock symbol
            force_download (bool): Force re-download even if cached data exists
            
        Returns:
            Optional[pd.DataFrame]: Processed stock data
        """
        logger.info(f"Processing data for {symbol}")
        
        # Check if processed data already exists
        processed_file = f"data/processed/model_inputs/{symbol}_daily.csv"
        if not force_download and os.path.exists(processed_file):
            try:
                df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded existing processed data for {symbol}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing data for {symbol}: {e}")
        
        # Download raw data using free tier endpoint
        raw_data = self.api_client.get_stock_data(symbol, "full")
        if raw_data is None:
            logger.error(f"Failed to download data for {symbol}")
            return None
        
        # Log information about data source
        logger.info(f"Downloaded {len(raw_data)} days of data for {symbol} using free tier endpoint")
        logger.debug(f"Data columns for {symbol}: {list(raw_data.columns)}")
        
        # Save raw data
        raw_file = f"data/raw/daily_prices/{symbol}_raw.csv"
        raw_data.to_csv(raw_file)
        logger.info(f"Saved raw data for {symbol} to {raw_file}")
        
        # Validate data
        validation_report = self.data_validator.validate_stock_data(raw_data, symbol)
        if not validation_report['is_valid']:
            logger.error(f"Data validation failed for {symbol}: {validation_report['issues']}")
            return None
        
        if validation_report['warnings']:
            logger.warning(f"Data validation warnings for {symbol}: {validation_report['warnings']}")
        
        # Filter by date range
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        filtered_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)]
        
        # Basic data cleaning
        cleaned_data = self._clean_data(filtered_data)
        
        # Add basic derived features
        enhanced_data = self._add_basic_features(cleaned_data)
        
        # Save processed data
        enhanced_data.to_csv(processed_file)
        logger.info(f"Saved processed data for {symbol} to {processed_file}")
        
        # Save data summary
        summary = self._get_data_summary(enhanced_data)
        summary_file = f"data/processed/features/{symbol}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return enhanced_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning."""
        df_clean = df.copy()
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Handle missing values (forward fill)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        return df_clean
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features."""
        df_enhanced = df.copy()
        
        # Ensure we have adjusted_close (using close as proxy for free tier)
        if 'adjusted_close' not in df_enhanced.columns and 'close' in df_enhanced.columns:
            df_enhanced['adjusted_close'] = df_enhanced['close'].copy()
            logger.info("Using close price as adjusted_close for free tier compatibility")
        
        # Daily returns
        if 'adjusted_close' in df_enhanced.columns:
            df_enhanced['daily_return'] = df_enhanced['adjusted_close'].pct_change()
            df_enhanced['log_return'] = np.log(df_enhanced['adjusted_close'] / df_enhanced['adjusted_close'].shift(1))
            df_enhanced['cumulative_return'] = (1 + df_enhanced['daily_return']).cumprod() - 1
        
        # Basic moving averages
        if 'adjusted_close' in df_enhanced.columns:
            for window in [5, 10, 20, 50]:
                df_enhanced[f'ma_{window}'] = df_enhanced['adjusted_close'].rolling(window=window).mean()
        
        # Price range
        if all(col in df_enhanced.columns for col in ['high', 'low', 'close']):
            df_enhanced['price_range'] = df_enhanced['high'] - df_enhanced['low']
            df_enhanced['price_range_pct'] = df_enhanced['price_range'] / df_enhanced['close']
        
        return df_enhanced
    
    def _get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate basic data summary."""
        summary = {
            'shape': df.shape,
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None,
                'total_days': len(df)
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_source': 'Alpha Vantage TIME_SERIES_DAILY (free tier)',
            'adjusted_close_note': 'Using close price as proxy for adjusted_close'
        }
        
        if 'adjusted_close' in df.columns:
            prices = df['adjusted_close'].dropna()
            if len(prices) > 1:
                returns = prices.pct_change().dropna()
                
                summary['price_analysis'] = {
                    'start_price': float(prices.iloc[0]),
                    'end_price': float(prices.iloc[-1]),
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'total_return': float((prices.iloc[-1] / prices.iloc[0]) - 1),
                    'volatility': float(returns.std() * np.sqrt(252)),
                    'avg_daily_return': float(returns.mean())
                }
        
        return summary
    
    def download_multiple_stocks(self, symbols: Optional[List[str]] = None, 
                                force_download: bool = False,
                                max_stocks_per_day: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Download and process data for multiple stocks with rate limiting.
        
        Args:
            symbols (Optional[List[str]]): List of stock symbols, uses config if None
            force_download (bool): Force re-download even if cached data exists
            max_stocks_per_day (Optional[int]): Maximum stocks to download per day
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of processed stock data
        """
        if symbols is None:
            symbols = self.config['data']['tickers']
        
        if max_stocks_per_day:
            symbols = symbols[:max_stocks_per_day]
        
        logger.info(f"Starting batch download for {len(symbols)} stocks: {symbols}")
        
        # Check remaining API calls
        remaining_calls = self.api_client.get_remaining_calls()
        logger.info(f"API calls remaining today: {remaining_calls}")
        
        if remaining_calls < len(symbols):
            logger.warning(f"Insufficient API calls remaining ({remaining_calls}) for all requested stocks ({len(symbols)})")
            logger.info(f"Will process up to {remaining_calls} stocks today")
            symbols = symbols[:remaining_calls]
        
        results = {}
        failed_downloads = []
        rate_limited = False
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            # Check API rate limit before each call
            if not self.api_client._check_rate_limit():
                logger.warning(f"Rate limit reached. Processed {len(results)} out of {len(symbols)} stocks.")
                rate_limited = True
                break
            
            try:
                data = self.download_single_stock(symbol, force_download)
                if data is not None:
                    results[symbol] = data
                    logger.info(f"Successfully processed {symbol} - {len(data)} days of data")
                else:
                    failed_downloads.append(symbol)
                    logger.warning(f"Failed to process {symbol}")
                
                # Add delay between requests to be respectful to API
                if i < len(symbols) - 1:
                    time.sleep(2)  # Increased delay for free tier
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_downloads.append(symbol)
        
        # Generate download report
        self._generate_download_report(symbols, results, failed_downloads, rate_limited)
        
        logger.info(f"Batch download completed. Successfully processed {len(results)} stocks.")
        if failed_downloads:
            logger.warning(f"Failed to process: {failed_downloads}")
        
        return results
    
    def _generate_download_report(self, requested_symbols: List[str], 
                                 successful_results: Dict[str, pd.DataFrame],
                                 failed_downloads: List[str],
                                 rate_limited: bool):
        """Generate comprehensive download report."""
        report = {
            'download_timestamp': datetime.now().isoformat(),
            'requested_symbols': requested_symbols,
            'successful_downloads': list(successful_results.keys()),
            'failed_downloads': failed_downloads,
            'rate_limited': rate_limited,
            'success_rate': len(successful_results) / len(requested_symbols) if requested_symbols else 0,
            'api_calls_remaining': self.api_client.get_remaining_calls(),
            'api_endpoint_used': 'TIME_SERIES_DAILY (free tier)',
            'data_notes': 'Using close price as proxy for adjusted_close',
            'data_summary': {}
        }
        
        # Add data summary for successful downloads
        for symbol, data in successful_results.items():
            if not data.empty:
                report['data_summary'][symbol] = {
                    'start_date': data.index.min().strftime('%Y-%m-%d'),
                    'end_date': data.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(data),
                    'columns': len(data.columns),
                    'missing_values': data.isnull().sum().sum()
                }
        
        # Save report
        report_file = f"data/processed/download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Download report saved to {report_file}")
    
    def get_data_status(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Get status of downloaded data for stocks.
        
        Args:
            symbols (Optional[List[str]]): List of symbols to check
            
        Returns:
            Dict[str, Dict]: Status information for each symbol
        """
        if symbols is None:
            symbols = self.config['data']['tickers']
        
        status = {}
        
        for symbol in symbols:
            processed_file = f"data/processed/model_inputs/{symbol}_daily.csv"
            raw_file = f"data/raw/daily_prices/{symbol}_raw.csv"
            
            symbol_status = {
                'symbol': symbol,
                'raw_data_exists': os.path.exists(raw_file),
                'processed_data_exists': os.path.exists(processed_file),
                'last_update': None,
                'data_range': None,
                'total_days': 0,
                'data_source': 'Alpha Vantage TIME_SERIES_DAILY (free tier)'
            }
            
            if os.path.exists(processed_file):
                try:
                    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                    symbol_status.update({
                        'last_update': df.index.max().strftime('%Y-%m-%d'),
                        'data_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
                        'total_days': len(df),
                        'file_size_mb': os.path.getsize(processed_file) / (1024 * 1024)
                    })
                except Exception as e:
                    symbol_status['error'] = str(e)
            
            status[symbol] = symbol_status
        
        return status
    
    def cleanup_old_cache(self, days_old: int = 7):
        """
        Clean up old cache files.
        
        Args:
            days_old (int): Remove cache files older than this many days
        """
        cache_dir = Path(self.config['data']['cache_dir'])
        if not cache_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_files = []
        
        for file_path in cache_dir.iterdir():
            if file_path.is_file():
                file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mod_time < cutoff_date:
                    try:
                        file_path.unlink()
                        cleaned_files.append(file_path.name)
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if cleaned_files:
            logger.info(f"Cleaned up {len(cleaned_files)} old cache files")
        else:
            logger.info("No old cache files found")
    
    def update_existing_data(self, symbol: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Update existing stock data with recent data.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back for updates
            
        Returns:
            Optional[pd.DataFrame]: Updated data
        """
        processed_file = f"data/processed/model_inputs/{symbol}_daily.csv"
        
        if not os.path.exists(processed_file):
            logger.info(f"No existing data for {symbol}, performing full download")
            return self.download_single_stock(symbol)
        
        try:
            existing_data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            last_date = existing_data.index.max()
            
            # Check if update is needed
            days_since_update = (datetime.now().date() - last_date.date()).days
            if days_since_update <= 1:
                logger.info(f"Data for {symbol} is up to date")
                return existing_data
            
            logger.info(f"Updating {symbol} data (last update: {last_date.date()})")
            
            # For simplicity with free tier limitations, just re-download and replace
            return self.download_single_stock(symbol, force_download=True)
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
            return self.download_single_stock(symbol, force_download=True)
    
    def export_data_for_analysis(self, symbols: Optional[List[str]] = None,
                                format: str = 'csv',
                                include_technical_indicators: bool = True) -> str:
        """
        Export processed data in various formats for external analysis.
        
        Args:
            symbols (Optional[List[str]]): Symbols to export
            format (str): Export format ('csv', 'excel', 'json')
            include_technical_indicators (bool): Whether to include technical indicators
            
        Returns:
            str: Path to exported file
        """
        if symbols is None:
            symbols = self.config['data']['tickers']
        
        export_data = {}
        
        for symbol in symbols:
            processed_file = f"data/processed/model_inputs/{symbol}_daily.csv"
            if os.path.exists(processed_file):
                try:
                    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                    
                    if not include_technical_indicators:
                        # Keep only basic OHLCV and derived price features
                        basic_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume',
                                       'daily_return', 'log_return', 'cumulative_return']
                        available_basic = [col for col in basic_columns if col in df.columns]
                        df = df[available_basic]
                    
                    export_data[symbol] = df
                    
                except Exception as e:
                    logger.warning(f"Failed to load data for {symbol}: {e}")
        
        # Create export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = "data/exports"
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            # Export each stock as separate CSV
            for symbol, df in export_data.items():
                export_file = f"{export_dir}/{symbol}_export_{timestamp}.csv"
                df.to_csv(export_file)
            
            export_path = f"{export_dir}/export_{timestamp}"
            logger.info(f"Exported {len(export_data)} stocks to CSV files in {export_path}")
            
        elif format == 'excel':
            export_file = f"{export_dir}/stocks_export_{timestamp}.xlsx"
            with pd.ExcelWriter(export_file, engine='openpyxl') as writer:
                for symbol, df in export_data.items():
                    # Excel sheet names have character limits
                    sheet_name = symbol[:31]  # Excel limit is 31 characters
                    df.to_excel(writer, sheet_name=sheet_name)
            
            export_path = export_file
            logger.info(f"Exported {len(export_data)} stocks to Excel file: {export_file}")
            
        elif format == 'json':
            export_file = f"{export_dir}/stocks_export_{timestamp}.json"
            
            # Convert dataframes to JSON-serializable format
            json_data = {}
            for symbol, df in export_data.items():
                json_data[symbol] = df.to_dict('index')
            
            with open(export_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            export_path = export_file
            logger.info(f"Exported {len(export_data)} stocks to JSON file: {export_file}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return export_path