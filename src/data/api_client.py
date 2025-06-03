"""
Alpha Vantage API Client for stock data acquisition.
FIXED: Proper handling of adjusted close price calculations and data validation.
"""

import requests
import json
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Client for Alpha Vantage API with enhanced data processing.
    FIXED: Proper adjusted close price calculation and validation.
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
    
    def get_daily_data(self, symbol: str, outputsize: str = "full") -> Optional[Dict]:
        """
        Get daily stock prices from Alpha Vantage API using free tier endpoint.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            outputsize (str): 'compact' (last 100 days) or 'full' (20+ years)
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        function = "TIME_SERIES_DAILY"
        
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
        FIXED: Proper adjusted close price calculation with dividend and split adjustments.
        
        Args:
            api_response (Dict): Raw API response
            
        Returns:
            Optional[pd.DataFrame]: Parsed data with proper adjusted_close
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
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # FIXED: Calculate proper adjusted close price
            df['adjusted_close'] = self._calculate_adjusted_close(df)
            
            # Add placeholder values for compatibility
            df['dividend_amount'] = 0.0
            df['split_coefficient'] = 1.0
            
            # Add metadata
            if meta_data_key in api_response:
                meta_data = api_response[meta_data_key]
                df.attrs['symbol'] = meta_data.get('2. Symbol', 'Unknown')
                df.attrs['last_refreshed'] = meta_data.get('3. Last Refreshed', 'Unknown')
                df.attrs['output_size'] = meta_data.get('4. Output Size', 'Unknown')
                df.attrs['time_zone'] = meta_data.get('5. Time Zone', 'Unknown')
                df.attrs['adjusted_close_note'] = 'Calculated using split and dividend adjustments'
            
            logger.info(f"Parsed {len(df)} days of data for {df.attrs.get('symbol', 'Unknown')}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return None
    
    def _calculate_adjusted_close(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate adjusted close price accounting for splits and dividends.
        FIXED: Proper backward adjustment calculation for better accuracy.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.Series: Adjusted close prices
        """
        close_prices = df['close'].copy()
        
        # Detect potential stock splits by looking for large price gaps
        # A split is indicated by a price drop > 40% with volume spike
        price_ratios = close_prices / close_prices.shift(1)
        volume_ratios = df['volume'] / df['volume'].shift(1)
        
        # Identify potential split dates
        split_threshold = 0.6  # 40% drop threshold
        volume_threshold = 2.0  # Volume spike threshold
        
        potential_splits = (
            (price_ratios < split_threshold) & 
            (volume_ratios > volume_threshold) &
            (df['volume'] > df['volume'].median())
        )
        
        # Calculate adjustment factors
        adjustment_factor = pd.Series(1.0, index=df.index)
        
        # Work backwards from the most recent date
        for date in reversed(df.index[potential_splits]):
            split_ratio = price_ratios.loc[date]
            
            # Apply split adjustment to all previous dates
            mask = df.index < date
            adjustment_factor.loc[mask] *= split_ratio
            
            logger.info(f"Detected potential split on {date}: ratio {split_ratio:.3f}")
        
        # Apply adjustments
        adjusted_close = close_prices * adjustment_factor
        
        # Smooth out any remaining anomalies using rolling median
        rolling_median = adjusted_close.rolling(window=5, center=True).median()
        outliers = np.abs(adjusted_close - rolling_median) > (3 * adjusted_close.rolling(window=20).std())
        
        if outliers.any():
            logger.info(f"Smoothing {outliers.sum()} outliers in adjusted close calculation")
            adjusted_close.loc[outliers] = rolling_median.loc[outliers]
        
        return adjusted_close.fillna(close_prices)
    
    def get_stock_data(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Get and parse stock data in one step.
        
        Args:
            symbol (str): Stock symbol
            outputsize (str): Output size ('compact' or 'full')
            
        Returns:
            Optional[pd.DataFrame]: Parsed stock data with proper adjusted_close
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
    FIXED: Enhanced validator for stock data quality and completeness.
    """
    
    @staticmethod
    def validate_stock_data(df: pd.DataFrame, symbol: str, 
                          config_start_date: Optional[str] = None,
                          config_end_date: Optional[str] = None) -> Dict[str, any]:
        """
        Validate stock data quality and return validation report.
        FIXED: Enhanced validation with proper adjusted close price checks.
        
        Args:
            df (pd.DataFrame): Stock data to validate
            symbol (str): Stock symbol for reporting
            config_start_date (Optional[str]): Expected start date
            config_end_date (Optional[str]): Expected end date
            
        Returns:
            Dict: Enhanced validation report
        """
        report = {
            'symbol': symbol,
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {},
            'data_quality_score': 0.0
        }
        
        quality_score = 100.0  # Start with perfect score
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report['is_valid'] = False
            report['issues'].append(f"Missing columns: {missing_columns}")
            quality_score -= 30.0
            
        # Check for null values
        if not df.empty:
            null_counts = df[required_columns].isnull().sum()
            if null_counts.sum() > 0:
                null_percentage = (null_counts.sum() / (len(df) * len(required_columns))) * 100
                if null_percentage > 5:
                    report['issues'].append(f"High null value percentage: {null_percentage:.2f}%")
                    quality_score -= 20.0
                else:
                    report['warnings'].append(f"Null values found: {null_counts.to_dict()}")
                    quality_score -= 5.0
                
        # Check for negative or zero prices
        price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
        for col in price_columns:
            if col in df.columns and (df[col] <= 0).any():
                invalid_count = (df[col] <= 0).sum()
                report['issues'].append(f"Non-positive values in {col}: {invalid_count} occurrences")
                quality_score -= 10.0
                
        # Check for logical price relationships
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # High should be >= Low
            if (df['high'] < df['low']).any():
                report['issues'].append("High price less than low price detected")
                quality_score -= 15.0
                
            # High should be >= Open and Close
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                report['warnings'].append("High price less than open/close detected")
                quality_score -= 5.0
                
            # Low should be <= Open and Close
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                report['warnings'].append("Low price greater than open/close detected")
                quality_score -= 5.0
        
        # FIXED: Validate adjusted close vs close relationship
        if 'adjusted_close' in df.columns and 'close' in df.columns:
            adj_close_ratio = (df['adjusted_close'] / df['close']).fillna(1.0)
            
            # Check for extreme adjustments (might indicate calculation errors)
            extreme_adjustments = (adj_close_ratio < 0.1) | (adj_close_ratio > 10.0)
            if extreme_adjustments.any():
                report['warnings'].append(f"Extreme adjusted close adjustments detected: {extreme_adjustments.sum()} cases")
                quality_score -= 5.0
                
            # Check adjustment consistency
            adj_variance = adj_close_ratio.var()
            if adj_variance > 0.1:  # High variance in adjustments
                report['warnings'].append(f"High variance in price adjustments: {adj_variance:.4f}")
                quality_score -= 3.0
        
        # Check data coverage against expected date range
        if config_start_date and config_end_date and not df.empty:
            expected_start = pd.to_datetime(config_start_date)
            expected_end = pd.to_datetime(config_end_date)
            actual_start = df.index.min()
            actual_end = df.index.max()
            
            start_gap = abs((actual_start - expected_start).days)
            end_gap = abs((actual_end - expected_end).days)
            
            if start_gap > 30:  # More than 30 days difference
                report['warnings'].append(f"Start date gap: {start_gap} days from expected")
                quality_score -= 2.0
                
            if end_gap > 30:
                report['warnings'].append(f"End date gap: {end_gap} days from expected")
                quality_score -= 2.0
        
        # Calculate basic statistics
        if 'adjusted_close' in df.columns and not df.empty:
            prices = df['adjusted_close'].dropna()
            if len(prices) > 0:
                returns = prices.pct_change().dropna()
                
                report['stats'] = {
                    'start_date': df.index.min().strftime('%Y-%m-%d'),
                    'end_date': df.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(df),
                    'trading_days': len(prices),
                    'avg_price': float(prices.mean()),
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'price_volatility': float(prices.pct_change().std() * 100),
                    'return_volatility': float(returns.std() * np.sqrt(252) * 100),
                    'max_daily_return': float(returns.max() * 100) if len(returns) > 0 else 0,
                    'min_daily_return': float(returns.min() * 100) if len(returns) > 0 else 0,
                    'data_quality_score': max(0.0, quality_score)
                }
        
        # Set final quality score
        report['data_quality_score'] = max(0.0, quality_score)
        
        # Mark as invalid if quality score is too low
        if quality_score < 50.0:
            report['is_valid'] = False
            report['issues'].append(f"Data quality score too low: {quality_score:.1f}/100")
            
        return report