"""
Logging Configuration Module for Stock Trading Optimization.
Provides structured logging with different levels and output formats.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for better readability.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        # Add color codes
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


class TradingLogger:
    """
    Centralized logging configuration for the trading optimization system.
    """
    
    def __init__(self, name: str = "trading_optimization"):
        """
        Initialize trading logger.
        
        Args:
            name (str): Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.handlers_configured = False
        
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_file: Optional[str] = None,
                     log_dir: str = "logs",
                     enable_console: bool = True,
                     enable_file: bool = True,
                     enable_json: bool = False,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5) -> logging.Logger:
        """
        Setup comprehensive logging configuration.
        
        Args:
            log_level (str): Logging level
            log_file (Optional[str]): Log file name
            log_dir (str): Log directory
            enable_console (bool): Enable console logging
            enable_file (bool): Enable file logging
            enable_json (bool): Enable JSON formatted logging
            max_file_size (int): Maximum log file size in bytes
            backup_count (int): Number of backup files to keep
            
        Returns:
            logging.Logger: Configured logger
        """
        # Convert string level to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Clear existing handlers if reconfiguring
        if self.handlers_configured:
            self.logger.handlers.clear()
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            
            console_formatter = ColoredConsoleFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            if log_file is None:
                log_file = f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_path = log_path / log_file
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if enable_json:
            json_file = log_path / f"{self.name}_structured.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(numeric_level)
            json_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(json_handler)
        
        self.handlers_configured = True
        
        # Log configuration info
        self.logger.info(f"Logging configured - Level: {log_level}, Console: {enable_console}, "
                        f"File: {enable_file}, JSON: {enable_json}")
        
        return self.logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        if not self.handlers_configured:
            self.setup_logging()
        return self.logger
    
    def log_function_call(self, func_name: str, args: tuple = (), 
                         kwargs: Dict[str, Any] = None):
        """
        Log function call with parameters.
        
        Args:
            func_name (str): Function name
            args (tuple): Function arguments
            kwargs (Dict[str, Any]): Function keyword arguments
        """
        kwargs = kwargs or {}
        
        # Sanitize arguments for logging
        safe_args = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                safe_args.append(arg)
            else:
                safe_args.append(type(arg).__name__)
        
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                safe_kwargs[key] = value
            else:
                safe_kwargs[key] = type(value).__name__
        
        self.logger.debug(f"Calling {func_name} with args={safe_args}, kwargs={safe_kwargs}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any], context: str = ""):
        """
        Log performance metrics in a structured way.
        
        Args:
            metrics (Dict[str, Any]): Performance metrics
            context (str): Context description
        """
        context_str = f" - {context}" if context else ""
        self.logger.info(f"Performance Metrics{context_str}")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")
    
    def log_trade_execution(self, trade_info: Dict[str, Any]):
        """
        Log trade execution details.
        
        Args:
            trade_info (Dict[str, Any]): Trade information
        """
        action = trade_info.get('action', 'unknown')
        symbol = trade_info.get('symbol', 'UNKNOWN')
        price = trade_info.get('price', 0)
        quantity = trade_info.get('quantity', 0)
        
        self.logger.info(f"Trade Executed - {action.upper()} {quantity} shares of {symbol} at ${price:.2f}")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """
        Log error with additional context.
        
        Args:
            error (Exception): Exception that occurred
            context (Dict[str, Any]): Additional context information
        """
        context = context or {}
        
        self.logger.error(f"Error occurred: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
        
        if context:
            self.logger.error(f"Context: {context}")
        
        # Log full stack trace at debug level
        self.logger.debug("Full stack trace:", exc_info=True)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        class_name = self.__class__.__name__
        return logging.getLogger(f"trading_optimization.{class_name}")
    
    def log_method_entry(self, method_name: str, **kwargs):
        """Log method entry with parameters."""
        params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.debug(f"Entering {method_name}({params})")
    
    def log_method_exit(self, method_name: str, result: Any = None):
        """Log method exit with result."""
        if result is not None:
            result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            self.logger.debug(f"Exiting {method_name} with result: {result_str}")
        else:
            self.logger.debug(f"Exiting {method_name}")
    
    def log_progress(self, current: int, total: int, description: str = ""):
        """Log progress updates."""
        percentage = (current / total) * 100 if total > 0 else 0
        desc_str = f" - {description}" if description else ""
        self.logger.info(f"Progress: {current}/{total} ({percentage:.1f}%){desc_str}")


def setup_global_logging(config: Dict[str, Any] = None) -> logging.Logger:
    """
    Setup global logging configuration from config dictionary.
    
    Args:
        config (Dict[str, Any]): Logging configuration
        
    Returns:
        logging.Logger: Root logger
    """
    if config is None:
        config = {
            'level': 'INFO',
            'enable_console': True,
            'enable_file': True,
            'log_dir': 'logs'
        }
    
    trading_logger = TradingLogger("trading_optimization")
    logger = trading_logger.setup_logging(**config)
    
    # Also configure the root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(getattr(logging, config.get('level', 'INFO').upper()))
        
        # Add a simple console handler to root logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name, if None uses the calling module name
        
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# Decorator for automatic function logging
def log_function_calls(logger_name: str = None):
    """
    Decorator to automatically log function calls.
    
    Args:
        logger_name (str): Logger name to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Log function entry
            func_name = f"{func.__module__}.{func.__name__}"
            logger.debug(f"Entering {func_name}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful exit
                logger.debug(f"Exiting {func_name} successfully")
                return result
                
            except Exception as e:
                # Log error
                logger.error(f"Error in {func_name}: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    config = {
        'log_level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'enable_json': True,
        'log_dir': 'test_logs'
    }
    
    logger = setup_global_logging(config)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    trading_logger = TradingLogger("test")
    trading_logger.log_performance_metrics({
        'total_return': 0.1523,
        'sharpe_ratio': 1.234,
        'max_drawdown': 0.087
    }, "Backtest Results")
    
    # Test trade logging
    trading_logger.log_trade_execution({
        'action': 'buy',
        'symbol': 'AAPL',
        'price': 150.25,
        'quantity': 100
    })
    
    print("Logging test completed. Check test_logs directory for output files.")