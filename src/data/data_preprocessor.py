"""
Data preprocessing module for stock trading optimization.
Handles data cleaning, feature engineering, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """
    Comprehensive data preprocessing for stock trading data.
    """
    
    def __init__(self, fill_method: str = 'forward', 
                 outlier_method: str = 'iqr',
                 min_trading_days: int = 252):
        """
        Initialize data preprocessor.
        
        Args:
            fill_method (str): Method for filling missing values ('forward', 'backward', 'interpolate')
            outlier_method (str): Method for outlier detection ('iqr', 'zscore', 'none')
            min_trading_days (int): Minimum number of trading days required
        """
        self.fill_method = fill_method
        self.outlier_method = outlier_method
        self.min_trading_days = min_trading_days
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info(f"Cleaning data with {len(df)} rows")
        
        # Make a copy
        df_clean = df.copy()
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicate dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Detect and handle outliers
        if self.outlier_method != 'none':
            df_clean = self._handle_outliers(df_clean)
        
        # Validate price relationships
        df_clean = self._validate_price_relationships(df_clean)
        
        # Check minimum trading days requirement
        if len(df_clean) < self.min_trading_days:
            logger.warning(f"Data has only {len(df_clean)} days, less than minimum {self.min_trading_days}")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            return df
        
        logger.info(f"Handling missing values: {missing_counts.to_dict()}")
        
        df_filled = df.copy()
        
        if self.fill_method == 'forward':
            df_filled = df_filled.fillna(method='ffill')
        elif self.fill_method == 'backward':
            df_filled = df_filled.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            df_filled = df_filled.interpolate(method='linear')
        
        # Fill any remaining NaN with the nearest valid value
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
        
        return df_filled
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data."""
        df_clean = df.copy()
        
        price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
        
        for col in price_columns:
            if col in df_clean.columns:
                if self.outlier_method == 'iqr':
                    df_clean = self._remove_outliers_iqr(df_clean, col)
                elif self.outlier_method == 'zscore':
                    df_clean = self._remove_outliers_zscore(df_clean, col)
        
        return df_clean
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str, 
                            multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        if outliers.sum() > 0:
            logger.info(f"Removing {outliers.sum()} outliers in {column} using IQR method")
            
            # Replace outliers with median value
            median_value = df[column].median()
            df.loc[outliers, column] = median_value
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, column: str, 
                               threshold: float = 3) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        
        if outliers.sum() > 0:
            logger.info(f"Removing {outliers.sum()} outliers in {column} using Z-score method")
            
            # Replace outliers with median value
            median_value = df[column].median()
            df.loc[outliers, column] = median_value
        
        return df
    
    def _validate_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix logical price relationships."""
        df_valid = df.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in df_valid.columns for col in price_cols):
            
            # Ensure High >= Low
            invalid_high_low = df_valid['high'] < df_valid['low']
            if invalid_high_low.sum() > 0:
                logger.warning(f"Found {invalid_high_low.sum()} rows where High < Low")
                # Swap values
                df_valid.loc[invalid_high_low, ['high', 'low']] = df_valid.loc[invalid_high_low, ['low', 'high']].values
            
            # Ensure High >= Open, Close and Low <= Open, Close
            for price_col in ['open', 'close']:
                # Fix High < price
                invalid_high = df_valid['high'] < df_valid[price_col]
                if invalid_high.sum() > 0:
                    logger.warning(f"Found {invalid_high.sum()} rows where High < {price_col}")
                    df_valid.loc[invalid_high, 'high'] = df_valid.loc[invalid_high, price_col]
                
                # Fix Low > price
                invalid_low = df_valid['low'] > df_valid[price_col]
                if invalid_low.sum() > 0:
                    logger.warning(f"Found {invalid_low.sum()} rows where Low > {price_col}")
                    df_valid.loc[invalid_low, 'low'] = df_valid.loc[invalid_low, price_col]
        
        return df_valid
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for analysis.
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        logger.info("Adding derived features")
        
        df_enhanced = df.copy()
        
        # Price-based features
        if 'adjusted_close' in df_enhanced.columns:
            # Daily returns
            df_enhanced['daily_return'] = df_enhanced['adjusted_close'].pct_change()
            
            # Log returns
            df_enhanced['log_return'] = np.log(df_enhanced['adjusted_close'] / df_enhanced['adjusted_close'].shift(1))
            
            # Cumulative returns
            df_enhanced['cumulative_return'] = (1 + df_enhanced['daily_return']).cumprod() - 1
        
        # OHLC-based features
        if all(col in df_enhanced.columns for col in ['open', 'high', 'low', 'close']):
            # Typical price
            df_enhanced['typical_price'] = (df_enhanced['high'] + df_enhanced['low'] + df_enhanced['close']) / 3
            
            # Price range
            df_enhanced['price_range'] = df_enhanced['high'] - df_enhanced['low']
            df_enhanced['price_range_pct'] = df_enhanced['price_range'] / df_enhanced['close']
            
            # Gap indicators
            df_enhanced['gap'] = df_enhanced['open'] - df_enhanced['close'].shift(1)
            df_enhanced['gap_pct'] = df_enhanced['gap'] / df_enhanced['close'].shift(1)
            
            # Intraday return
            df_enhanced['intraday_return'] = (df_enhanced['close'] - df_enhanced['open']) / df_enhanced['open']
        
        # Volume-based features (if available)
        if 'volume' in df_enhanced.columns:
            # Volume moving averages
            df_enhanced['volume_ma_5'] = df_enhanced['volume'].rolling(window=5).mean()
            df_enhanced['volume_ma_20'] = df_enhanced['volume'].rolling(window=20).mean()
            
            # Volume ratio
            df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_ma_20']
            
            # Volume-price trend
            if 'adjusted_close' in df_enhanced.columns:
                df_enhanced['vpt'] = (df_enhanced['daily_return'] * df_enhanced['volume']).cumsum()
        
        # Rolling statistics
        if 'adjusted_close' in df_enhanced.columns:
            windows = [5, 10, 20, 50]
            for window in windows:
                # Rolling mean
                df_enhanced[f'price_ma_{window}'] = df_enhanced['adjusted_close'].rolling(window=window).mean()
                
                # Rolling standard deviation
                df_enhanced[f'price_std_{window}'] = df_enhanced['adjusted_close'].rolling(window=window).std()
                
                # Rolling min/max
                df_enhanced[f'price_min_{window}'] = df_enhanced['adjusted_close'].rolling(window=window).min()
                df_enhanced[f'price_max_{window}'] = df_enhanced['adjusted_close'].rolling(window=window).max()
                
                # Position within range
                df_enhanced[f'price_position_{window}'] = ((df_enhanced['adjusted_close'] - df_enhanced[f'price_min_{window}']) / 
                                                          (df_enhanced[f'price_max_{window}'] - df_enhanced[f'price_min_{window}']))
        
        logger.info(f"Added derived features. New shape: {df_enhanced.shape}")
        return df_enhanced
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               train_ratio: float = 0.7,
                               validation_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits for time series data.
        
        Args:
            df (pd.DataFrame): Input data
            train_ratio (float): Ratio for training data
            validation_ratio (float): Ratio for validation data
            test_ratio (float): Ratio for test data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
        """
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, "
                   f"Validation: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def normalize_data(self, df: pd.DataFrame, 
                      method: str = 'minmax',
                      columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize specified columns of the data.
        
        Args:
            df (pd.DataFrame): Input data
            method (str): Normalization method ('minmax', 'zscore', 'robust')
            columns (Optional[List[str]]): Columns to normalize, if None normalizes numeric columns
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Normalized data and scaling parameters
        """
        df_norm = df.copy()
        scaling_params = {}
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df_norm.columns:
                if method == 'minmax':
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                    scaling_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                    
                elif method == 'zscore':
                    mean_val = df_norm[col].mean()
                    std_val = df_norm[col].std()
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
                    scaling_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                    
                elif method == 'robust':
                    median_val = df_norm[col].median()
                    mad_val = (df_norm[col] - median_val).abs().median()
                    df_norm[col] = (df_norm[col] - median_val) / mad_val
                    scaling_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df_norm, scaling_params
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data summary.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict: Data summary statistics
        """
        summary = {
            'shape': df.shape,
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None,
                'total_days': len(df)
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['statistics'] = df[numeric_cols].describe().to_dict()
        
        # Price-specific analysis
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
                    'avg_daily_return': float(returns.mean()),
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
        
        return summary


class FeatureEngineer:
    """
    Advanced feature engineering for stock trading models.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df (pd.DataFrame): Input data
            columns (List[str]): Columns to create lags for
            lags (List[int]): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lagged features
        """
        df_lagged = df.copy()
        
        for col in columns:
            if col in df_lagged.columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    df_lagged[lag_col_name] = df_lagged[col].shift(lag)
                    self.feature_names.append(lag_col_name)
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame,
                               column: str,
                               windows: List[int],
                               functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df (pd.DataFrame): Input data
            column (str): Column to compute rolling features for
            windows (List[int]): List of window sizes
            functions (List[str]): List of functions to apply
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        df_rolling = df.copy()
        
        if column not in df_rolling.columns:
            return df_rolling
        
        for window in windows:
            for func in functions:
                feature_name = f"{column}_rolling_{func}_{window}"
                
                if func == 'mean':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).mean()
                elif func == 'std':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).std()
                elif func == 'min':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).min()
                elif func == 'max':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).max()
                elif func == 'skew':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).skew()
                elif func == 'kurt':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).kurt()
                
                self.feature_names.append(feature_name)
        
        return df_rolling
    
    def create_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary technical trading signals.
        
        Args:
            df (pd.DataFrame): Input data with price columns
            
        Returns:
            pd.DataFrame: Data with technical signals
        """
        df_signals = df.copy()
        
        if 'adjusted_close' in df_signals.columns:
            price = df_signals['adjusted_close']
            
            # Moving average signals
            for short, long in [(5, 20), (10, 50), (20, 200)]:
                short_ma = price.rolling(window=short).mean()
                long_ma = price.rolling(window=long).mean()
                
                signal_name = f"ma_signal_{short}_{long}"
                df_signals[signal_name] = (short_ma > long_ma).astype(int)
                self.feature_names.append(signal_name)
            
            # Price vs MA signals
            for window in [10, 20, 50]:
                ma = price.rolling(window=window).mean()
                signal_name = f"price_above_ma_{window}"
                df_signals[signal_name] = (price > ma).astype(int)
                self.feature_names.append(signal_name)
            
            # Momentum signals
            for period in [5, 10, 20]:
                momentum = price / price.shift(period) - 1
                signal_name = f"momentum_positive_{period}"
                df_signals[signal_name] = (momentum > 0).astype(int)
                self.feature_names.append(signal_name)
        
        return df_signals