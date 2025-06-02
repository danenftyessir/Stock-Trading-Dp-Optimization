"""
Alpha Vantage API Client for stock data acquisition.
Fixed to use free tier endpoints only.
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Client for Alpha Vantage API with rate limiting and caching capabilities.
    Updated to use free tier endpoints.
    """
    
    def __init__(self, api_key: str, cache_dir: str = "data/raw/api_cache",
                 rate_limit: int = 25):
        """
        Initialize Alpha Vantage API client.
        
        Args:
            api_key (str): Alpha Vantage API key
            cache_dir (str): Directory for caching API responses
            rate_limit (int): Maximum API calls per day
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.calls_made_today = 0
        self.last_call_date = None
        
        # Load call history
        self._load_call_history()
        
    def _load_call_history(self):
        """Load API call history from cache."""
        history_file = self.cache_dir / "call_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    today = datetime.now().strftime("%Y-%m-%d")
                    if history.get('date') == today:
                        self.calls_made_today = history.get('calls', 0)
                        self.last_call_date = today
            except Exception as e:
                logger.warning(f"Could not load call history: {e}")
                
    def _save_call_history(self):
        """Save API call history to cache."""
        history_file = self.cache_dir / "call_history.json"
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Reset counter if it's a new day
        if self.last_call_date != today:
            self.calls_made_today = 0
            self.last_call_date = today
            
        history = {
            'date': today,
            'calls': self.calls_made_today
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.warning(f"Could not save call history: {e}")
            
    def _check_rate_limit(self) -> bool:
        """Check if we can make another API call today."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Reset counter if it's a new day
        if self.last_call_date != today:
            self.calls_made_today = 0
            self.last_call_date = today
            
        return self.calls_made_today < self.rate_limit
    
    def _get_cache_filename(self, symbol: str, function: str) -> Path:
        """Generate cache filename for API response."""
        return self.cache_dir / f"{symbol}_{function}_{datetime.now().strftime('%Y%m%d')}.json"
    
    def _load_from_cache(self, symbol: str, function: str) -> Optional[Dict]:
        """Load data from cache if available and recent."""
        cache_file = self._get_cache_filename(symbol, function)
        
        # Also check for older cache files (within last 7 days)
        for days_back in range(7):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            old_cache_file = self.cache_dir / f"{symbol}_{function}_{date_str}.json"
            
            if old_cache_file.exists():
                try:
                    with open(old_cache_file, 'r') as f:
                        data = json.load(f)
                        logger.info(f"Loaded {symbol} data from cache: {old_cache_file}")
                        return data
                except Exception as e:
                    logger.warning(f"Could not load cache file {old_cache_file}: {e}")
                    
        return None
    
    def _save_to_cache(self, symbol: str, function: str, data: Dict):
        """Save API response to cache."""
        cache_file = self._get_cache_filename(symbol, function)
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {symbol} data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save to cache: {e}")
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = "full") -> Optional[Dict]:
        """
        Get daily adjusted stock prices from Alpha Vantage API using free tier endpoint.
        Note: This method now uses TIME_SERIES_DAILY (free tier) instead of TIME_SERIES_DAILY_ADJUSTED.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            outputsize (str): 'compact' (last 100 days) or 'full' (20+ years)
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        return self.get_daily_data(symbol, outputsize)
    
    def get_daily_data(self, symbol: str, outputsize: str = "full") -> Optional[Dict]:
        """
        Get daily stock prices from Alpha Vantage API using free tier endpoint.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            outputsize (str): 'compact' (last 100 days) or 'full' (20+ years)
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        function = "TIME_SERIES_DAILY"  # Free tier endpoint
        
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol, function)
        if cached_data:
            return cached_data
            
        # Check rate limit
        if not self._check_rate_limit():
            logger.error(f"Rate limit exceeded ({self.rate_limit} calls/day). "
                        f"Calls made today: {self.calls_made_today}")
            return None
            
        # Prepare API request
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        
        try:
            logger.info(f"Making API call for {symbol} (call {self.calls_made_today + 1}/{self.rate_limit})")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None
                
            if 'Information' in data:
                logger.warning(f"API Information: {data['Information']}")
                # This might be a rate limit message
                return None
                
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
            
            # Check if we got valid time series data
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
            
            # Update call counter
            self.calls_made_today += 1
            self._save_call_history()
            
            # Save to cache
            self._save_to_cache(symbol, function, data)
            
            # Add delay to be respectful to API
            time.sleep(1)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            return None
    
    def parse_daily_data(self, api_response: Dict) -> Optional[pd.DataFrame]:
        """
        Parse Alpha Vantage daily data response into pandas DataFrame.
        Updated to handle free tier TIME_SERIES_DAILY format.
        
        Args:
            api_response (Dict): Raw API response
            
        Returns:
            Optional[pd.DataFrame]: Parsed data with OHLCV columns
        """
        try:
            time_series_key = "Time Series (Daily)"
            meta_data_key = "Meta Data"
            
            if time_series_key not in api_response:
                logger.error("Time series data not found in API response")
                return None
                
            time_series = api_response[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Clean column names and convert to numeric
            # Free tier TIME_SERIES_DAILY format:
            # "1. open", "2. high", "3. low", "4. close", "5. volume"
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # For compatibility, use close price as adjusted_close
            df['adjusted_close'] = df['close'].copy()
            
            # Add dummy values for missing fields (since we don't have adjusted data)
            df['dividend_amount'] = 0.0
            df['split_coefficient'] = 1.0
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 
                             'volume', 'dividend_amount', 'split_coefficient']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Add metadata
            if meta_data_key in api_response:
                meta_data = api_response[meta_data_key]
                df.attrs['symbol'] = meta_data.get('2. Symbol', 'Unknown')
                df.attrs['last_refreshed'] = meta_data.get('3. Last Refreshed', 'Unknown')
                df.attrs['output_size'] = meta_data.get('4. Output Size', 'Unknown')
                df.attrs['time_zone'] = meta_data.get('5. Time Zone', 'Unknown')
            
            logger.info(f"Parsed {len(df)} days of data for {df.attrs.get('symbol', 'Unknown')}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return None
    
    def get_stock_data(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Get and parse stock data in one step.
        
        Args:
            symbol (str): Stock symbol
            outputsize (str): Output size ('compact' or 'full')
            
        Returns:
            Optional[pd.DataFrame]: Parsed stock data
        """
        raw_data = self.get_daily_data(symbol, outputsize)
        if raw_data:
            return self.parse_daily_data(raw_data)
        return None
    
    def get_multiple_stocks(self, symbols: List[str], outputsize: str = "full") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks with rate limiting.
        
        Args:
            symbols (List[str]): List of stock symbols
            outputsize (str): Output size for each request
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching data for {symbol} ({i+1}/{len(symbols)})")
            
            data = self.get_stock_data(symbol, outputsize)
            if data is not None:
                results[symbol] = data
                logger.info(f"Successfully fetched {len(data)} days for {symbol}")
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
                
            # Check if we've hit rate limit
            if not self._check_rate_limit():
                logger.warning(f"Rate limit reached. Successfully fetched {len(results)} out of {len(symbols)} stocks.")
                break
                
        return results
    
    def get_remaining_calls(self) -> int:
        """Get number of API calls remaining for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_call_date != today:
            return self.rate_limit
        return max(0, self.rate_limit - self.calls_made_today)


class DataValidator:
    """
    Validator for stock data quality and completeness.
    """
    
    @staticmethod
    def validate_stock_data(df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Validate stock data quality and return validation report.
        
        Args:
            df (pd.DataFrame): Stock data to validate
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Validation report
        """
        report = {
            'symbol': symbol,
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report['is_valid'] = False
            report['issues'].append(f"Missing columns: {missing_columns}")
            
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            report['warnings'].append(f"Null values found: {null_counts.to_dict()}")
            
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
        for col in price_columns:
            if col in df.columns and (df[col] <= 0).any():
                report['warnings'].append(f"Non-positive values in {col}")
                
        # Check for logical price relationships
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # High should be >= Low
            if (df['high'] < df['low']).any():
                report['issues'].append("High price less than low price detected")
                
            # High should be >= Open and Close
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                report['warnings'].append("High price less than open/close detected")
                
            # Low should be <= Open and Close
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                report['warnings'].append("Low price greater than open/close detected")
        
        # Calculate basic statistics
        if 'adjusted_close' in df.columns:
            prices = df['adjusted_close'].dropna()
            if len(prices) > 0:
                report['stats'] = {
                    'start_date': df.index.min().strftime('%Y-%m-%d'),
                    'end_date': df.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(df),
                    'trading_days': len(prices),
                    'avg_price': float(prices.mean()),
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'volatility': float(prices.pct_change().std() * 100)
                }
        
        if report['issues']:
            report['is_valid'] = False
            
        return report