"""
Performance Metrics and Risk Analysis Module.
FIXED VERSION - Corrected win rate calculation and removed artificial metric capping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    FIXED: Accurate metric calculations without artificial constraints.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate (float): Annual risk-free rate
            trading_days (int): Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
    def calculate_returns(self, portfolio_values: List[float]) -> np.ndarray:
        """
        Calculate daily returns from portfolio values.
        FIXED: Proper handling of edge cases without artificial constraints.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            np.ndarray: Daily returns
        """
        if len(portfolio_values) < 2:
            return np.array([])
            
        portfolio_array = np.array(portfolio_values)
        
        # Handle any non-positive values
        portfolio_array = np.maximum(portfolio_array, 1e-8)
        
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        
        return returns
    
    def total_return(self, initial_value: float, final_value: float) -> float:
        """Calculate total return."""
        if initial_value <= 0:
            return 0.0
        return (final_value - initial_value) / initial_value
    
    def annualized_return(self, portfolio_values: List[float]) -> float:
        """
        Calculate annualized return.
        FIXED: Accurate calculation without artificial capping.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            float: Annualized return
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        num_days = len(portfolio_values) - 1
        
        if num_days == 0 or initial_value <= 0 or final_value <= 0:
            return 0.0
        
        # Calculate years
        years = num_days / self.trading_days
        
        if years <= 0:
            return 0.0
            
        try:
            annualized = ((final_value / initial_value) ** (1 / years)) - 1
            return annualized if np.isfinite(annualized) else 0.0
        except (OverflowError, ZeroDivisionError):
            return 0.0
    
    def volatility(self, portfolio_values: List[float]) -> float:
        """
        Calculate annualized volatility.
        FIXED: Accurate volatility calculation without artificial limits.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            float: Annualized volatility
        """
        returns = self.calculate_returns(portfolio_values)
        if len(returns) == 0:
            return 0.0
        
        volatility = np.std(returns) * np.sqrt(self.trading_days)
        
        return volatility if np.isfinite(volatility) else 0.0
    
    def sharpe_ratio(self, portfolio_values: List[float]) -> float:
        """
        Calculate Sharpe ratio.
        FIXED: Proper Sharpe ratio calculation without artificial capping.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            float: Sharpe ratio
        """
        returns = self.calculate_returns(portfolio_values)
        if len(returns) == 0:
            return 0.0
        
        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / self.trading_days
        
        # Calculate excess returns
        excess_returns = returns - daily_rf_rate
        
        mean_excess_return = np.mean(excess_returns)
        std_returns = np.std(returns)
        
        if std_returns == 0 or np.isnan(std_returns) or np.isinf(std_returns):
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe = mean_excess_return / std_returns * np.sqrt(self.trading_days)
        
        return sharpe if np.isfinite(sharpe) else 0.0
    
    def sortino_ratio(self, portfolio_values: List[float]) -> float:
        """
        Calculate Sortino ratio (like Sharpe but using downside deviation).
        FIXED: Accurate calculation without artificial limits.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            float: Sortino ratio
        """
        returns = self.calculate_returns(portfolio_values)
        if len(returns) == 0:
            return 0.0
        
        daily_rf_rate = self.risk_free_rate / self.trading_days
        excess_returns = returns - daily_rf_rate
        mean_excess_return = np.mean(excess_returns)
        
        downside_returns = returns[returns < daily_rf_rate]
        
        if len(downside_returns) == 0:
            # No downside risk - return high but finite value
            return 10.0 if mean_excess_return > 0 else 0.0
        
        downside_deviation = np.std(downside_returns) * np.sqrt(self.trading_days)
        
        if downside_deviation == 0:
            return 10.0 if mean_excess_return > 0 else 0.0
        
        sortino = (mean_excess_return * self.trading_days) / downside_deviation
        
        return sortino if np.isfinite(sortino) else 0.0
    
    def max_drawdown(self, portfolio_values: List[float]) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            Dict: Maximum drawdown metrics
        """
        if len(portfolio_values) < 2:
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'recovery_time': 0}
            
        portfolio_array = np.array(portfolio_values)
        portfolio_array = np.maximum(portfolio_array, 1e-8)  # Avoid division by zero
        
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        
        max_dd = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        # Find drawdown duration and recovery time
        peak_idx = np.argmax(portfolio_array[:max_dd_idx + 1])
        
        # Recovery time (days to recover from max drawdown)
        recovery_idx = max_dd_idx
        peak_value = portfolio_array[peak_idx]
        
        for i in range(max_dd_idx + 1, len(portfolio_array)):
            if portfolio_array[i] >= peak_value:
                recovery_idx = i
                break
        
        drawdown_duration = max_dd_idx - peak_idx
        recovery_time = recovery_idx - max_dd_idx
        
        return {
            'max_drawdown': max_dd,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time,
            'peak_value': peak_value,
            'trough_value': portfolio_array[max_dd_idx]
        }
    
    def calmar_ratio(self, portfolio_values: List[float]) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        FIXED: Accurate calculation without artificial limits.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            
        Returns:
            float: Calmar ratio
        """
        annual_return = self.annualized_return(portfolio_values)
        max_dd = self.max_drawdown(portfolio_values)['max_drawdown']
        
        if max_dd == 0:
            return 10.0 if annual_return > 0 else 0.0
        
        calmar = annual_return / max_dd
        
        return calmar if np.isfinite(calmar) else 0.0
    
    def value_at_risk(self, portfolio_values: List[float], 
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            confidence_level (float): Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            float: VaR value
        """
        returns = self.calculate_returns(portfolio_values)
        if len(returns) == 0:
            return 0.0
            
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def conditional_var(self, portfolio_values: List[float], 
                       confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR or Expected Shortfall).
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            confidence_level (float): Confidence level
            
        Returns:
            float: CVaR value
        """
        returns = self.calculate_returns(portfolio_values)
        if len(returns) == 0:
            return 0.0
            
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
            
        return abs(np.mean(tail_losses))
    
    def information_ratio(self, portfolio_values: List[float], 
                         benchmark_values: List[float]) -> float:
        """
        Calculate Information Ratio.
        FIXED: Accurate calculation without artificial limits.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            benchmark_values (List[float]): Benchmark values
            
        Returns:
            float: Information ratio
        """
        if len(portfolio_values) != len(benchmark_values) or len(portfolio_values) < 2:
            return 0.0
            
        portfolio_returns = self.calculate_returns(portfolio_values)
        benchmark_returns = self.calculate_returns(benchmark_values)
        
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        excess_returns = portfolio_returns - benchmark_returns
        
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0 or np.isnan(tracking_error):
            return 0.0
            
        info_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(self.trading_days)
        
        return info_ratio if np.isfinite(info_ratio) else 0.0
    
    def beta(self, portfolio_values: List[float], 
            benchmark_values: List[float]) -> float:
        """
        Calculate portfolio beta relative to benchmark.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            benchmark_values (List[float]): Benchmark values
            
        Returns:
            float: Beta coefficient
        """
        if len(portfolio_values) != len(benchmark_values) or len(portfolio_values) < 2:
            return 0.0
            
        portfolio_returns = self.calculate_returns(portfolio_values)
        benchmark_returns = self.calculate_returns(benchmark_values)
        
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance == 0 or np.isnan(benchmark_variance):
                return 0.0
                
            beta_val = covariance / benchmark_variance
            
            return beta_val if np.isfinite(beta_val) else 1.0
            
        except:
            return 1.0
    
    def alpha(self, portfolio_values: List[float], 
             benchmark_values: List[float]) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            benchmark_values (List[float]): Benchmark values
            
        Returns:
            float: Alpha value
        """
        portfolio_return = self.annualized_return(portfolio_values)
        benchmark_return = self.annualized_return(benchmark_values)
        portfolio_beta = self.beta(portfolio_values, benchmark_values)
        
        expected_return = self.risk_free_rate + portfolio_beta * (benchmark_return - self.risk_free_rate)
        
        alpha_val = portfolio_return - expected_return
        
        return alpha_val if np.isfinite(alpha_val) else 0.0
    
    def win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate win rate from trades.
        FIXED: Accurate win rate calculation based on individual trade profits.
        
        Args:
            trades (List[Dict]): List of trade dictionaries with 'profit' key
            
        Returns:
            float: Win rate (percentage of profitable trades)
        """
        if not trades:
            return 0.0
        
        # Count profitable trades more accurately
        profitable_trades = 0
        total_valid_trades = 0
        
        for trade in trades:
            profit = trade.get('profit', 0)
            
            # Skip trades with invalid or missing profit data
            if profit is None or not np.isfinite(profit):
                continue
                
            total_valid_trades += 1
            
            # A trade is profitable if profit > small threshold (to account for rounding)
            if profit > 1e-6:  # Small positive threshold
                profitable_trades += 1
        
        if total_valid_trades == 0:
            return 0.0
            
        win_rate = profitable_trades / total_valid_trades
        
        # Log suspicious win rates for debugging
        if win_rate == 1.0 and total_valid_trades > 5:
            logger.warning(f"Suspicious 100% win rate detected with {total_valid_trades} trades. "
                          f"This may indicate perfect hindsight optimization.")
        
        return win_rate
    
    def comprehensive_analysis(self, portfolio_values: List[float],
                             trades: Optional[List[Dict]] = None,
                             benchmark_values: Optional[List[float]] = None,
                             dates: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive performance analysis.
        FIXED: Accurate analysis with proper validation and no artificial constraints.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            trades (Optional[List[Dict]]): Trade details
            benchmark_values (Optional[List[float]]): Benchmark comparison
            dates (Optional[List[str]]): Date labels
            
        Returns:
            Dict: Comprehensive performance metrics
        """
        logger.info("Performing comprehensive performance analysis")
        
        if len(portfolio_values) < 2:
            logger.warning("Insufficient data for performance analysis")
            return self._empty_metrics()
        
        # Basic return metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        if initial_value <= 0 or final_value <= 0:
            logger.warning("Invalid portfolio values detected")
            return self._empty_metrics()
        
        metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': self.total_return(initial_value, final_value),
            'annualized_return': self.annualized_return(portfolio_values),
            'volatility': self.volatility(portfolio_values),
        }
        
        # Risk-adjusted metrics
        metrics.update({
            'sharpe_ratio': self.sharpe_ratio(portfolio_values),
            'sortino_ratio': self.sortino_ratio(portfolio_values),
            'calmar_ratio': self.calmar_ratio(portfolio_values),
        })
        
        # Drawdown analysis
        dd_metrics = self.max_drawdown(portfolio_values)
        metrics.update({
            'max_drawdown': dd_metrics['max_drawdown'],
            'drawdown_duration': dd_metrics['drawdown_duration'],
            'recovery_time': dd_metrics['recovery_time'],
        })
        
        # Risk metrics
        metrics.update({
            'value_at_risk_95': self.value_at_risk(portfolio_values, 0.95),
            'conditional_var_95': self.conditional_var(portfolio_values, 0.95),
            'value_at_risk_99': self.value_at_risk(portfolio_values, 0.99),
            'conditional_var_99': self.conditional_var(portfolio_values, 0.99),
        })
        
        # Trade-based metrics
        if trades:
            metrics.update({
                'number_of_trades': len(trades),
                'win_rate': self.win_rate(trades),
                'profit_factor': self.profit_factor(trades),
                'average_trade_profit': self.average_trade_profit(trades),
            })
            
            win_loss = self.largest_win_loss(trades)
            metrics.update(win_loss)
        
        # Benchmark comparison
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            metrics.update({
                'beta': self.beta(portfolio_values, benchmark_values),
                'alpha': self.alpha(portfolio_values, benchmark_values),
                'information_ratio': self.information_ratio(portfolio_values, benchmark_values),
                'benchmark_total_return': self.total_return(benchmark_values[0], benchmark_values[-1]),
                'excess_return': metrics['total_return'] - self.total_return(benchmark_values[0], benchmark_values[-1]),
            })
        
        # Time-based metrics
        if dates and len(dates) == len(portfolio_values):
            try:
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[-1])
                metrics.update({
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'total_days': (end_date - start_date).days,
                    'trading_days': len(portfolio_values) - 1,
                })
            except:
                pass
        
        logger.info("Performance analysis completed")
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics for invalid data."""
        return {
            'initial_value': 0.0,
            'final_value': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'drawdown_duration': 0,
            'recovery_time': 0,
            'value_at_risk_95': 0.0,
            'conditional_var_95': 0.0,
            'value_at_risk_99': 0.0,
            'conditional_var_99': 0.0,
        }
    
    def profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Args:
            trades (List[Dict]): List of trade dictionaries with 'profit' key
            
        Returns:
            float: Profit factor
        """
        if not trades:
            return 0.0
            
        gross_profits = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
        gross_losses = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
        
        if gross_losses == 0:
            return 10.0 if gross_profits > 0 else 0.0
            
        profit_factor = gross_profits / gross_losses
        
        return profit_factor if np.isfinite(profit_factor) else 0.0
    
    def average_trade_profit(self, trades: List[Dict]) -> float:
        """Calculate average profit per trade."""
        if not trades:
            return 0.0
        
        valid_profits = [trade.get('profit', 0) for trade in trades if np.isfinite(trade.get('profit', 0))]
        
        if not valid_profits:
            return 0.0
            
        return sum(valid_profits) / len(valid_profits)
    
    def largest_win_loss(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Find largest winning and losing trades.
        
        Args:
            trades (List[Dict]): List of trade dictionaries
            
        Returns:
            Dict: Largest win and loss amounts
        """
        if not trades:
            return {'largest_win': 0.0, 'largest_loss': 0.0}
            
        profits = [trade.get('profit', 0) for trade in trades if np.isfinite(trade.get('profit', 0))]
        
        if not profits:
            return {'largest_win': 0.0, 'largest_loss': 0.0}
        
        return {
            'largest_win': max(profits),
            'largest_loss': min(profits)
        }


class RiskAnalyzer:
    """
    Specialized risk analysis tools.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        self.confidence_levels = confidence_levels