"""
Baseline Trading Strategies for comparison with Dynamic Programming approach.
FIXED: Resolved ImportError by using absolute imports and fallback import handling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# FIXED: Import handling to avoid relative import issues
try:
    # Try absolute import first
    from analysis.performance_metrics import PerformanceAnalyzer
except ImportError:
    try:
        # Try relative import as fallback
        from ..analysis.performance_metrics import PerformanceAnalyzer
    except ImportError:
        # If both fail, use None and implement fallback metrics
        PerformanceAnalyzer = None
        logger.warning("PerformanceAnalyzer not available, using fallback metrics calculation")


class BuyAndHoldStrategy:
    """
    Simple Buy and Hold strategy for baseline comparison.
    FIXED: Enhanced implementation with proper transaction cost handling.
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize Buy and Hold strategy.
        
        Args:
            transaction_cost (float): Transaction cost as percentage
        """
        self.transaction_cost = transaction_cost
        self.name = "Buy & Hold Strategy"
        
    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """
        Execute Buy and Hold strategy.
        FIXED: More realistic execution with proper validation.
        
        Args:
            prices (List[float]): Historical stock prices
            initial_capital (float): Initial capital
            
        Returns:
            Dict: Strategy execution results
        """
        if len(prices) < 2:
            return self._empty_result(initial_capital)
        
        # Validate price data
        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected in Buy & Hold strategy")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < 2:
                return self._empty_result(initial_capital)
            prices = prices_array[valid_indices].tolist()
            
        # Buy at first price, sell at last price
        buy_price = prices[0]
        sell_price = prices[-1]
        
        # Calculate shares that can be bought
        shares = initial_capital / (buy_price * (1 + self.transaction_cost))
        
        # Calculate final value after selling
        final_value = shares * sell_price * (1 - self.transaction_cost)
        
        # Calculate portfolio values over time
        portfolio_values = []
        for price in prices:
            portfolio_values.append(shares * price)
            
        total_return = (final_value - initial_capital) / initial_capital
        
        # Create trade record
        trades = [{
            'buy_day': 0,
            'buy_price': buy_price,
            'sell_day': len(prices) - 1,
            'sell_price': sell_price,
            'profit': final_value - initial_capital,
            'shares': shares,
            'duration': len(prices) - 1
        }]
        
        # Calculate metrics using available analyzer or fallback
        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed: {e}, using fallback")
                metrics = self._calculate_basic_metrics(portfolio_values, trades, initial_capital)
        else:
            metrics = self._calculate_basic_metrics(portfolio_values, trades, initial_capital)
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'num_transactions': 1,
            'max_profit': final_value - initial_capital,
            'metrics': metrics
        }
    
    def _calculate_basic_metrics(self, portfolio_values: List[float], 
                                trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate basic metrics as fallback."""
        if not portfolio_values:
            return {}
        
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        returns = returns[np.isfinite(returns)]
        
        metrics = {
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
            'sharpe_ratio': (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': 1.0 if trades and trades[0].get('profit', 0) > 0 else 0.0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)
    
    def _empty_result(self, initial_capital: float) -> Dict:
        """Return empty result for invalid inputs."""
        return {
            'strategy': self.name,
            'total_return': 0.0,
            'final_value': initial_capital,
            'portfolio_values': [initial_capital],
            'trades': [],
            'num_transactions': 0,
            'max_profit': 0.0,
            'metrics': {'total_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        }


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover strategy.
    FIXED: Enhanced implementation with better signal generation and validation.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50,
                 transaction_cost: float = 0.001):
        """
        Initialize MA Crossover strategy.
        
        Args:
            short_window (int): Short moving average window
            long_window (int): Long moving average window
            transaction_cost (float): Transaction cost as percentage
        """
        self.short_window = short_window
        self.long_window = long_window
        self.transaction_cost = transaction_cost
        self.name = f"MA Crossover ({short_window}/{long_window})"
        
    def calculate_moving_averages(self, prices: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate short and long moving averages."""
        prices_array = np.array(prices)
        
        short_ma = pd.Series(prices_array).rolling(window=self.short_window).mean().values
        long_ma = pd.Series(prices_array).rolling(window=self.long_window).mean().values
        
        return short_ma, long_ma
    
    def generate_signals(self, prices: List[float]) -> List[str]:
        """
        Generate buy/sell/hold signals based on MA crossover.
        FIXED: Enhanced signal generation with better validation.
        
        Returns:
            List[str]: Signal for each day ('buy', 'sell', 'hold')
        """
        short_ma, long_ma = self.calculate_moving_averages(prices)
        signals = ['hold'] * len(prices)
        
        # Track position to avoid multiple buys/sells
        position = 'out'  # 'in' or 'out'
        
        for i in range(1, len(prices)):
            if (not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]) and
                not np.isnan(short_ma[i-1]) and not np.isnan(long_ma[i-1])):
                
                # Buy signal: short MA crosses above long MA and not already holding
                if (short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1] and 
                    position == 'out'):
                    signals[i] = 'buy'
                    position = 'in'
                    
                # Sell signal: short MA crosses below long MA and currently holding
                elif (short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1] and 
                      position == 'in'):
                    signals[i] = 'sell'
                    position = 'out'
                    
        return signals
    
    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """
        Execute Moving Average Crossover strategy.
        FIXED: More realistic execution with proper position tracking.
        
        Args:
            prices (List[float]): Historical stock prices
            initial_capital (float): Initial capital
            
        Returns:
            Dict: Strategy execution results
        """
        if len(prices) < max(self.short_window, self.long_window) + 1:
            return self._empty_result(initial_capital)
        
        # Validate price data
        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected in MA Crossover strategy")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < max(self.short_window, self.long_window) + 1:
                return self._empty_result(initial_capital)
            # For simplicity, keep original prices but handle them in signal generation
            
        signals = self.generate_signals(prices)
        
        # Simulate trading with enhanced logic
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 'out'  # 'in' or 'out' of market
        
        for day, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 'buy' and position == 'out' and cash > 0:
                # Buy shares
                trade_cost = price * (1 + self.transaction_cost)
                shares_to_buy = cash / trade_cost
                cost = shares_to_buy * trade_cost
                
                if cost <= cash and shares_to_buy > 0:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'in'
                    
                    trades.append({
                        'action': 'buy',
                        'day': day,
                        'price': price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
                    
            elif signal == 'sell' and position == 'in' and shares > 0:
                # Sell shares
                proceeds = shares * price * (1 - self.transaction_cost)
                cash += proceeds
                
                trades.append({
                    'action': 'sell',
                    'day': day,
                    'price': price,
                    'shares': shares,
                    'proceeds': proceeds
                })
                
                shares = 0
                position = 'out'
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        # Final liquidation if still holding shares
        if shares > 0:
            final_proceeds = shares * prices[-1] * (1 - self.transaction_cost)
            cash += final_proceeds
            trades.append({
                'action': 'sell',
                'day': len(prices) - 1,
                'price': prices[-1],
                'shares': shares,
                'proceeds': final_proceeds
            })
            shares = 0
        
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        
        # Process trades into buy-sell pairs
        processed_trades = self._process_trades(trades)
        
        # Calculate metrics
        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=processed_trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed: {e}, using fallback")
                metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)
        else:
            metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': processed_trades,
            'num_transactions': len(processed_trades),
            'max_profit': final_value - initial_capital,
            'signals': signals,
            'metrics': metrics
        }
    
    def _process_trades(self, raw_trades: List[Dict]) -> List[Dict]:
        """Process raw trades into buy-sell pairs."""
        processed = []
        buy_trade = None
        
        for trade in raw_trades:
            if trade['action'] == 'buy':
                buy_trade = trade
            elif trade['action'] == 'sell' and buy_trade is not None:
                profit = trade['proceeds'] - buy_trade['cost']
                processed.append({
                    'buy_day': buy_trade['day'],
                    'buy_price': buy_trade['price'],
                    'sell_day': trade['day'],
                    'sell_price': trade['price'],
                    'profit': profit,
                    'shares': trade['shares'],
                    'duration': trade['day'] - buy_trade['day']
                })
                buy_trade = None
                
        return processed
    
    def _calculate_basic_metrics(self, portfolio_values: List[float], 
                                trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate basic metrics as fallback."""
        if not portfolio_values:
            return {}
        
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        returns = returns[np.isfinite(returns)]
        
        # Calculate win rate
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            win_rate = profitable_trades / len(trades)
        else:
            win_rate = 0.0
        
        metrics = {
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital if portfolio_values else 0.0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
            'sharpe_ratio': (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': win_rate
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)
    
    def _empty_result(self, initial_capital: float) -> Dict:
        """Return empty result for invalid inputs."""
        return {
            'strategy': self.name,
            'total_return': 0.0,
            'final_value': initial_capital,
            'portfolio_values': [initial_capital],
            'trades': [],
            'num_transactions': 0,
            'max_profit': 0.0,
            'signals': [],
            'metrics': {'total_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        }


class MomentumStrategy:
    """
    Simple momentum strategy based on recent price movements.
    FIXED: Enhanced momentum calculation with better signal filtering.
    """
    
    def __init__(self, lookback_window: int = 10, threshold: float = 0.02,
                 transaction_cost: float = 0.001):
        """
        Initialize Momentum strategy.
        
        Args:
            lookback_window (int): Number of days to look back for momentum
            threshold (float): Minimum return threshold to trigger signal
            transaction_cost (float): Transaction cost as percentage
        """
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.name = f"Momentum ({lookback_window}d, {threshold:.1%})"
        
    def calculate_momentum(self, prices: List[float]) -> List[float]:
        """
        Calculate momentum indicator.
        FIXED: Enhanced momentum calculation with better validation.
        """
        momentum = [0.0] * len(prices)
        
        for i in range(self.lookback_window, len(prices)):
            if prices[i - self.lookback_window] > 0:  # Avoid division by zero
                past_price = prices[i - self.lookback_window]
                current_price = prices[i]
                momentum[i] = (current_price - past_price) / past_price
            
        return momentum
    
    def generate_signals(self, prices: List[float]) -> List[str]:
        """
        Generate trading signals based on momentum.
        FIXED: Enhanced signal generation with position tracking.
        """
        momentum = self.calculate_momentum(prices)
        signals = ['hold'] * len(prices)
        
        position = 'out'  # Track position to avoid multiple entries
        
        for i, mom in enumerate(momentum):
            if mom > self.threshold and position == 'out':
                signals[i] = 'buy'
                position = 'in'
            elif mom < -self.threshold and position == 'in':
                signals[i] = 'sell'
                position = 'out'
            # Also sell if momentum becomes negative while holding
            elif mom < 0 and position == 'in':
                signals[i] = 'sell'
                position = 'out'
                
        return signals
    
    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """
        Execute momentum strategy.
        FIXED: Enhanced execution with better position management.
        """
        if len(prices) < self.lookback_window + 1:
            return self._empty_result(initial_capital)
        
        # Validate price data
        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected in Momentum strategy")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < self.lookback_window + 1:
                return self._empty_result(initial_capital)
                
        signals = self.generate_signals(prices)
        
        # Similar execution logic as MovingAverageCrossoverStrategy
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 'out'
        
        for day, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 'buy' and position == 'out' and cash > 0:
                trade_cost = price * (1 + self.transaction_cost)
                shares_to_buy = cash / trade_cost
                cost = shares_to_buy * trade_cost
                
                if cost <= cash and shares_to_buy > 0:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'in'
                    
                    trades.append({
                        'action': 'buy',
                        'day': day,
                        'price': price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
                    
            elif signal == 'sell' and position == 'in' and shares > 0:
                proceeds = shares * price * (1 - self.transaction_cost)
                cash += proceeds
                
                trades.append({
                    'action': 'sell',
                    'day': day,
                    'price': price,
                    'shares': shares,
                    'proceeds': proceeds
                })
                
                shares = 0
                position = 'out'
            
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        # Final liquidation
        if shares > 0:
            final_proceeds = shares * prices[-1] * (1 - self.transaction_cost)
            cash += final_proceeds
            trades.append({
                'action': 'sell',
                'day': len(prices) - 1,
                'price': prices[-1],
                'shares': shares,
                'proceeds': final_proceeds
            })
        
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        processed_trades = self._process_trades(trades)
        
        # Calculate metrics
        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=processed_trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed: {e}, using fallback")
                metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)
        else:
            metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': processed_trades,
            'num_transactions': len(processed_trades),
            'max_profit': final_value - initial_capital,
            'signals': signals,
            'metrics': metrics
        }
    
    def _process_trades(self, raw_trades: List[Dict]) -> List[Dict]:
        """Process raw trades into buy-sell pairs."""
        processed = []
        buy_trade = None
        
        for trade in raw_trades:
            if trade['action'] == 'buy':
                buy_trade = trade
            elif trade['action'] == 'sell' and buy_trade is not None:
                profit = trade['proceeds'] - buy_trade['cost']
                processed.append({
                    'buy_day': buy_trade['day'],
                    'buy_price': buy_trade['price'],
                    'sell_day': trade['day'],
                    'sell_price': trade['price'],
                    'profit': profit,
                    'shares': trade['shares'],
                    'duration': trade['day'] - buy_trade['day']
                })
                buy_trade = None
                
        return processed
    
    def _calculate_basic_metrics(self, portfolio_values: List[float], 
                                trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate basic metrics as fallback."""
        if not portfolio_values:
            return {}
        
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        returns = returns[np.isfinite(returns)]
        
        # Calculate win rate
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            win_rate = profitable_trades / len(trades)
        else:
            win_rate = 0.0
        
        metrics = {
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital if portfolio_values else 0.0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
            'sharpe_ratio': (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': win_rate
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)
    
    def _empty_result(self, initial_capital: float) -> Dict:
        """Return empty result for invalid inputs."""
        return {
            'strategy': self.name,
            'total_return': 0.0,
            'final_value': initial_capital,
            'portfolio_values': [initial_capital],
            'trades': [],
            'num_transactions': 0,
            'max_profit': 0.0,
            'signals': [],
            'metrics': {'total_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        }


class StrategyComparator:
    """
    Compare multiple trading strategies on the same data.
    FIXED: Enhanced comparison with better result formatting and validation.
    """
    
    def __init__(self, strategies: List, initial_capital: float = 100000):
        """
        Initialize strategy comparator.
        
        Args:
            strategies (List): List of strategy instances
            initial_capital (float): Initial capital for all strategies
        """
        self.strategies = strategies
        self.initial_capital = initial_capital
        
    def compare(self, prices: pd.Series, dates: Optional[pd.DatetimeIndex] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Compare all strategies on given price data.
        FIXED: Enhanced comparison with better error handling and result formatting.
        
        Args:
            prices (pd.Series): Historical stock prices
            dates (Optional[pd.DatetimeIndex]): Date index for the prices
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Comparison results table and individual strategy results
        """
        results = []
        individual_results = {}
        
        # Convert Series to list for strategy execution
        price_list = prices.tolist() if hasattr(prices, 'tolist') else list(prices)
        
        for strategy in self.strategies:
            try:
                result = strategy.execute(price_list, self.initial_capital)
                
                # Extract metrics - handle both direct metrics and nested metrics structure
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    metrics = result['metrics']
                else:
                    # Use the result itself if it contains metric fields
                    metrics = result
                
                # Calculate additional metrics if needed
                portfolio_values = result.get('portfolio_values', [])
                
                if portfolio_values:
                    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
                    returns = returns[np.isfinite(returns)]
                else:
                    returns = np.array([])
                
                # Compile comprehensive metrics
                strategy_metrics = {
                    'Strategy': result.get('strategy', strategy.name if hasattr(strategy, 'name') else 'Unknown'),
                    'Total Return': metrics.get('total_return', result.get('total_return', 0)),
                    'Final Value': result.get('final_value', portfolio_values[-1] if portfolio_values else self.initial_capital),
                    'Max Profit': result.get('max_profit', 0),
                    'Num Transactions': result.get('num_transactions', len(result.get('trades', []))),
                    'Annualized Return': self._calculate_annualized_return(portfolio_values),
                    'Volatility': metrics.get('volatility', np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', self._calculate_sharpe_ratio(returns)),
                    'Max Drawdown': metrics.get('max_drawdown', self._calculate_max_drawdown(portfolio_values)),
                    'Win Rate': metrics.get('win_rate', self._calculate_win_rate(result.get('trades', [])))
                }
                
                results.append(strategy_metrics)
                individual_results[strategy_metrics['Strategy']] = result
                
            except Exception as e:
                logger.error(f"Error executing {getattr(strategy, 'name', 'Unknown')}: {e}")
                
                # Add error entry
                error_metrics = {
                    'Strategy': getattr(strategy, 'name', 'Unknown'),
                    'Total Return': 0.0,
                    'Final Value': self.initial_capital,
                    'Max Profit': 0.0,
                    'Num Transactions': 0,
                    'Annualized Return': 0.0,
                    'Volatility': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Max Drawdown': 0.0,
                    'Win Rate': 0.0
                }
                results.append(error_metrics)
                individual_results[error_metrics['Strategy']] = {'error': str(e)}
                
        return pd.DataFrame(results), individual_results
    
    def _calculate_annualized_return(self, portfolio_values: List[float]) -> float:
        """Calculate annualized return."""
        if len(portfolio_values) < 2:
            return 0.0
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        years = len(portfolio_values) / 252  # Assume 252 trading days per year
        
        if years <= 0:
            return 0.0
        
        try:
            annualized = ((1 + total_return) ** (1 / years)) - 1
            return annualized if np.isfinite(annualized) else 0.0
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        return sharpe if np.isfinite(sharpe) else 0.0
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return profitable_trades / len(trades)