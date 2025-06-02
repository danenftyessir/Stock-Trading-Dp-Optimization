"""
Helper Utility Functions for Stock Trading Optimization.
Common utilities used across the trading optimization system.
"""

import os
import json
import pickle
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import hashlib
import functools
import time
import warnings


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path (str): Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize object to JSON-compatible format.
    
    Args:
        obj (Any): Object to serialize
        
    Returns:
        Any: JSON-serializable object
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return str(obj)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file with safe serialization.
    
    Args:
        data (Any): Data to save
        file_path (str): Output file path
        indent (int): JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = safe_json_serialize(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path (str): JSON file path
        
    Returns:
        Any: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object using pickle.
    
    Args:
        obj (Any): Object to save
        file_path (str): Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path (str): Pickle file path
        
    Returns:
        Any: Loaded object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path (str): Path to file
        algorithm (str): Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        str: File hash
    """
    hash_algo = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
    
    return hash_algo.hexdigest()


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency.
    
    Args:
        amount (float): Amount to format
        currency (str): Currency code
        
    Returns:
        str: Formatted currency string
    """
    if currency.upper() == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value (float): Value to format (0.1 = 10%)
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_trading_days(start_date: datetime, end_date: datetime,
                          weekends: bool = False, holidays: List[datetime] = None) -> int:
    """
    Calculate number of trading days between dates.
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        weekends (bool): Include weekends
        holidays (List[datetime]): List of holiday dates to exclude
        
    Returns:
        int: Number of trading days
    """
    if holidays is None:
        holidays = []
    
    current_date = start_date
    trading_days = 0
    
    while current_date <= end_date:
        # Skip weekends if not included
        if not weekends and current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Skip holidays
        if current_date.date() in [h.date() for h in holidays]:
            current_date += timedelta(days=1)
            continue
        
        trading_days += 1
        current_date += timedelta(days=1)
    
    return trading_days


def validate_data_range(data: pd.DataFrame, start_date: str, end_date: str) -> bool:
    """
    Validate that data covers the specified date range.
    
    Args:
        data (pd.DataFrame): Data with datetime index
        start_date (str): Required start date
        end_date (str): Required end date
        
    Returns:
        bool: True if data covers the range
    """
    if data.empty:
        return False
    
    data_start = data.index.min()
    data_end = data.index.max()
    
    required_start = pd.to_datetime(start_date)
    required_end = pd.to_datetime(end_date)
    
    return data_start <= required_start and data_end >= required_end


def resample_to_frequency(data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """
    Resample data to specified frequency.
    
    Args:
        data (pd.DataFrame): Input data with datetime index
        frequency (str): Target frequency ('D', 'W', 'M', etc.)
        
    Returns:
        pd.DataFrame: Resampled data
    """
    if frequency == 'D':
        return data  # Already daily
    
    # Define aggregation rules for different column types
    agg_rules = {}
    for col in data.columns:
        if 'price' in col.lower() or col.lower() in ['open', 'high', 'low', 'close', 'adjusted_close']:
            if col.lower() in ['open']:
                agg_rules[col] = 'first'
            elif col.lower() in ['high']:
                agg_rules[col] = 'max'
            elif col.lower() in ['low']:
                agg_rules[col] = 'min'
            else:
                agg_rules[col] = 'last'
        elif 'volume' in col.lower():
            agg_rules[col] = 'sum'
        else:
            agg_rules[col] = 'last'  # Default for other columns
    
    return data.resample(frequency).agg(agg_rules)


def detect_outliers(data: pd.Series, method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a data series.
    
    Args:
        data (pd.Series): Input data
        method (str): Detection method ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Function wrapper that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def memory_usage_decorator(func):
    """
    Decorator to measure memory usage of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Function wrapper that logs memory usage
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            print(f"{func.__name__} memory usage: {mem_diff:.2f} MB (before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)")
            
            return result
            
        except ImportError:
            warnings.warn("psutil not available, memory monitoring disabled")
            return func(*args, **kwargs)
    
    return wrapper


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      exceptions: Tuple = (Exception,)):
    """
    Decorator to retry function on exception.
    
    Args:
        max_retries (int): Maximum number of retries
        delay (float): Delay between retries in seconds
        exceptions (Tuple): Exception types to catch
        
    Returns:
        Function decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            
            return None
        
        return wrapper
    return decorator


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst (List[Any]): List to chunk
        chunk_size (int): Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d (Dict[str, Any]): Dictionary to flatten
        separator (str): Separator for nested keys
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(d)


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: Human-readable file size
    """
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if valid symbol format
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-5 uppercase letters
    return len(symbol) <= 5 and symbol.isalpha() and symbol.isupper()


def business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate business days between two dates.
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        int: Number of business days
    """
    return len(pd.bdate_range(start_date, end_date))


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """
    Create directory structure from nested dictionary.
    
    Args:
        base_path (str): Base directory path
        structure (Dict[str, Any]): Directory structure specification
    """
    base = Path(base_path)
    
    def create_dirs(current_path: Path, struct: Dict[str, Any]):
        for name, content in struct.items():
            new_path = current_path / name
            new_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, dict):
                create_dirs(new_path, content)
    
    create_dirs(base, structure)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = {}
    
    for config in configs:
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    
    return merged


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        if self.total == 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f", ETA: {eta:.0f}s" if eta > 0 else ""
        else:
            eta_str = ""
        
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%){eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


# Example usage and testing
if __name__ == "__main__":
    # Test configuration functions
    test_config = {
        'api': {
            'key': 'test_key',
            'rate_limit': 25
        },
        'data': {
            'symbols': ['AAPL', 'GOOGL'],
            'start_date': '2020-01-01'
        }
    }
    
    # Test save/load config
    save_config(test_config, 'test_config.yaml')
    loaded_config = load_config('test_config.yaml')
    print("Config test passed:", test_config == loaded_config)
    
    # Test JSON serialization
    test_data = {
        'timestamp': datetime.now(),
        'array': np.array([1, 2, 3]),
        'series': pd.Series([1, 2, 3])
    }
    
    save_json(test_data, 'test_data.json')
    print("JSON serialization test completed")
    
    # Test progress tracker
    tracker = ProgressTracker(100, "Test Progress")
    for i in range(100):
        time.sleep(0.01)  # Simulate work
        tracker.update()
    
    print("Helper functions test completed")