"""
Baseline Trading Strategies for comparison with Dynamic Programming approach.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from analysis.performance_metrics import PerformanceAnalyzer
except ImportError:
    try:
        from ..analysis.performance_metrics import PerformanceAnalyzer
    except ImportError:
        PerformanceAnalyzer = None
        logger.warning("PerformanceAnalyzer not available, using fallback metrics calculation")


class BuyAndHoldStrategy:
    """
    Simple Buy and Hold strategy for baseline comparison.
    """

    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize Buy and Hold strategy.

        Args:
            transaction_cost (float): Transaction cost as percentage
        """
        self.transaction_cost = transaction_cost
        self.name = "Buy & Hold Strategy"

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the strategy. Buy & Hold does not require fitting.
        Args:
            train_data (pd.DataFrame): Training data (not used).
        """
        pass # Strategi ini tidak memerlukan fitting

    def get_name(self) -> str:
        """
        Get the name of the strategy.
        Returns:
            str: Name of the strategy.
        """
        return self.name

    def predict_signals(self, data: pd.DataFrame) -> List[str]:
        """
        Generate trading signals for Buy & Hold.
        Args:
            data (pd.DataFrame): Price data.
        Returns:
            List[str]: List of signals ('buy', 'sell', 'hold').
        """
        n = len(data)
        if n < 1:
            return []
        signals = ['hold'] * n
        if n > 0:
            signals[0] = 'buy'
        if n > 1: # Ensure there's a day to sell
            signals[-1] = 'sell'
        return signals

    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """
        Execute Buy and Hold strategy.

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
            if len(prices) < 2: # Check again after filtering
                return self._empty_result(initial_capital)


        # Buy at first price, sell at last price
        buy_price = prices[0]
        sell_price = prices[-1]
        
        shares = 0
        if buy_price * (1 + self.transaction_cost) > 0: # Avoid division by zero
            shares = initial_capital / (buy_price * (1 + self.transaction_cost))

        # Calculate final value after selling
        final_value = shares * sell_price * (1 - self.transaction_cost)

        # Calculate portfolio values over time
        portfolio_values = []
        for price_point in prices: # Renamed variable to avoid conflict
            portfolio_values.append(shares * price_point)

        total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0


        # Create trade record
        trades = []
        if shares > 0 : # Only record trade if shares were bought
            trades.append({
                'buy_day': 0,
                'buy_price': buy_price,
                'sell_day': len(prices) - 1,
                'sell_price': sell_price,
                'profit': final_value - initial_capital,
                'shares': shares,
                'duration': len(prices) - 1
            })

        # Calculate metrics using available analyzer or fallback
        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed for BuyAndHold: {e}, using fallback")
                metrics = self._calculate_basic_metrics(portfolio_values, trades, initial_capital)
        else:
            metrics = self._calculate_basic_metrics(portfolio_values, trades, initial_capital)

        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'num_transactions': 1 if trades else 0, # Buy and Hold is considered 1 round trip transaction
            'max_profit': final_value - initial_capital, # For B&H, max_profit is the total profit
            'metrics': metrics
        }

    def _calculate_basic_metrics(self, portfolio_values: List[float],
                                trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate basic metrics as fallback."""
        if not portfolio_values or initial_capital <= 0:
            return self._empty_metrics_structure(initial_capital if initial_capital > 0 else 0.0)

        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1]) if len(portfolio_values) > 1 else np.array([])
        returns = returns[np.isfinite(returns)] 

        total_return_val = (portfolio_values[-1] - initial_capital) / initial_capital
        volatility_val = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        sharpe_ratio_val = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            daily_rf_rate = 0.02 / 252
            excess_returns = returns - daily_rf_rate
            sharpe_ratio_val = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        metrics = {
            'total_return': total_return_val,
            'annualized_return': ((portfolio_values[-1] / initial_capital) ** (252 / len(portfolio_values)) - 1) if len(portfolio_values) > 1 else 0.0,
            'volatility': volatility_val if np.isfinite(volatility_val) else 0.0,
            'sharpe_ratio': sharpe_ratio_val if np.isfinite(sharpe_ratio_val) else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': 1.0 if trades and trades[0].get('profit', 0) > 0 else 0.0,
            'initial_value': initial_capital,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'sortino_ratio': 0.0, 
            'calmar_ratio': 0.0,
            'drawdown_duration': 0,
            'recovery_time': 0,
            'value_at_risk_95': 0.0,
            'conditional_var_95': 0.0,
        }
        return metrics

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0

        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        peak[peak == 0] = 1e-8 
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0
        
    def _empty_result(self, initial_capital: float) -> Dict:
        """Return empty result for invalid inputs."""
        return {
            'strategy': self.name,
            'total_return': 0.0,
            'final_value': initial_capital,
            'portfolio_values': [initial_capital] if initial_capital > 0 else [0.0],
            'trades': [],
            'num_transactions': 0,
            'max_profit': 0.0,
            'metrics': self._empty_metrics_structure(initial_capital)
        }

    def _empty_metrics_structure(self, initial_capital = 0.0) -> Dict:
        """Returns a dictionary with all expected metric keys set to default values."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'number_of_trades': 0,
            'win_rate': 0.0,
            'initial_value': initial_capital,
            'final_value': initial_capital,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'drawdown_duration': 0,
            'recovery_time': 0,
            'value_at_risk_95': 0.0,
            'conditional_var_95': 0.0,
        }


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover strategy.
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

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the strategy. MA Crossover does not require fitting in this context.
        Args:
            train_data (pd.DataFrame): Training data (not used).
        """
        pass # Strategi ini tidak memerlukan fitting

    def get_name(self) -> str:
        """
        Get the name of the strategy.
        Returns:
            str: Name of the strategy.
        """
        return self.name

    def calculate_moving_averages(self, prices: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate short and long moving averages."""
        prices_array = np.array(prices)

        short_ma = pd.Series(prices_array).rolling(window=self.short_window).mean().values
        long_ma = pd.Series(prices_array).rolling(window=self.long_window).mean().values

        return short_ma, long_ma

    def predict_signals(self, data: pd.DataFrame) -> List[str]:
        """
        Generate trading signals based on MA crossover.
        Args:
            data (pd.DataFrame): Price data, expects 'adjusted_close' or 'close' column.
        Returns:
            List[str]: List of signals ('buy', 'sell', 'hold').
        """
        if 'adjusted_close' in data.columns:
            prices = data['adjusted_close'].tolist()
        elif 'close' in data.columns:
            prices = data['close'].tolist()
        else:
            logger.error("Price data ('adjusted_close' or 'close') not found for MA Crossover signals.")
            return ['hold'] * len(data)
            
        if len(prices) < self.long_window: # Ensure enough data for longest MA
             return ['hold'] * len(prices)

        short_ma, long_ma = self.calculate_moving_averages(prices)
        signals = ['hold'] * len(prices)
        position = 'out'

        for i in range(1, len(prices)):
            if (not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]) and
                not np.isnan(short_ma[i-1]) and not np.isnan(long_ma[i-1])):
                if (short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1] and position == 'out'):
                    signals[i] = 'buy'
                    position = 'in'
                elif (short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1] and position == 'in'):
                    signals[i] = 'sell'
                    position = 'out'
        return signals

    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """
        Execute Moving Average Crossover strategy.

        Args:
            prices (List[float]): Historical stock prices
            initial_capital (float): Initial capital

        Returns:
            Dict: Strategy execution results
        """
        if len(prices) < max(self.short_window, self.long_window) + 1:
            return self._empty_result(initial_capital)

        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected in MA Crossover strategy")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < max(self.short_window, self.long_window) + 1:
                return self._empty_result(initial_capital)
            prices = prices_array[valid_indices].tolist()
            if len(prices) < max(self.short_window, self.long_window) + 1:
                 return self._empty_result(initial_capital)


        # Generate signals using the internal method that expects a list of prices
        signals = self.predict_signals(pd.DataFrame({'close': prices}))


        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades_raw = [] 
        position = 'out'

        for day, (price_point, signal) in enumerate(zip(prices, signals)): 
            if signal == 'buy' and position == 'out' and cash > 0:
                trade_cost = price_point * (1 + self.transaction_cost)
                if trade_cost == 0: 
                    shares_to_buy = 0
                else:
                    shares_to_buy = cash / trade_cost
                
                cost = shares_to_buy * trade_cost

                if cost <= cash and shares_to_buy > 0:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'in'
                    trades_raw.append({
                        'action': 'buy', 'day': day, 'price': price_point,
                        'shares': shares_to_buy, 'cost': cost
                    })
            elif signal == 'sell' and position == 'in' and shares > 0:
                proceeds = shares * price_point * (1 - self.transaction_cost)
                cash += proceeds
                trades_raw.append({
                    'action': 'sell', 'day': day, 'price': price_point,
                    'shares': shares, 'proceeds': proceeds
                })
                shares = 0
                position = 'out'
            
            portfolio_value = cash + shares * price_point
            portfolio_values.append(portfolio_value)

        if shares > 0 and prices: 
            final_proceeds = shares * prices[-1] * (1 - self.transaction_cost)
            cash += final_proceeds
            trades_raw.append({
                'action': 'sell', 'day': len(prices) - 1, 'price': prices[-1],
                'shares': shares, 'proceeds': final_proceeds
            })
            shares = 0

        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        processed_trades = self._process_trades(trades_raw)

        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=processed_trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed for MA Crossover: {e}, using fallback")
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
        processed = []
        buy_trade = None
        for trade in raw_trades:
            if trade['action'] == 'buy':
                buy_trade = trade
            elif trade['action'] == 'sell' and buy_trade is not None:
                profit = trade['proceeds'] - buy_trade['cost']
                processed.append({
                    'buy_day': buy_trade['day'], 'buy_price': buy_trade['price'],
                    'sell_day': trade['day'], 'sell_price': trade['price'],
                    'profit': profit, 'shares': buy_trade['shares'], # Use shares from buy_trade
                    'duration': trade['day'] - buy_trade['day']
                })
                buy_trade = None
        return processed

    def _calculate_basic_metrics(self, portfolio_values: List[float],
                                trades: List[Dict], initial_capital: float) -> Dict:
        if not portfolio_values or initial_capital <= 0:
            return self._empty_metrics_structure(initial_capital if initial_capital > 0 else 0.0)
        
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1]) if len(portfolio_values) > 1 else np.array([])
        returns = returns[np.isfinite(returns)]

        win_rate_val = 0.0
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            if len(trades) > 0:
                 win_rate_val = profitable_trades / len(trades)

        total_return_val = (portfolio_values[-1] - initial_capital) / initial_capital
        volatility_val = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        sharpe_ratio_val = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            daily_rf_rate = 0.02 / 252
            excess_returns = returns - daily_rf_rate
            sharpe_ratio_val = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

        metrics = {
            'total_return': total_return_val,
            'annualized_return': ((portfolio_values[-1] / initial_capital) ** (252 / len(portfolio_values)) - 1) if len(portfolio_values) > 1 else 0.0,
            'volatility': volatility_val if np.isfinite(volatility_val) else 0.0,
            'sharpe_ratio': sharpe_ratio_val if np.isfinite(sharpe_ratio_val) else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': win_rate_val,
            'initial_value': initial_capital,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'drawdown_duration': 0,
            'recovery_time': 0, 'value_at_risk_95': 0.0, 'conditional_var_95': 0.0,
        }
        return metrics

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        if not portfolio_values: return 0.0
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        peak[peak == 0] = 1e-8
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def _empty_result(self, initial_capital: float) -> Dict:
        return {
            'strategy': self.name, 'total_return': 0.0, 'final_value': initial_capital,
            'portfolio_values': [initial_capital] if initial_capital > 0 else [0.0],
            'trades': [], 'num_transactions': 0, 'max_profit': 0.0, 'signals': [],
            'metrics': self._empty_metrics_structure(initial_capital)
        }

    def _empty_metrics_structure(self, initial_capital = 0.0) -> Dict:
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'number_of_trades': 0,
            'win_rate': 0.0, 'initial_value': initial_capital, 'final_value': initial_capital,
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'drawdown_duration': 0,
            'recovery_time': 0, 'value_at_risk_95': 0.0, 'conditional_var_95': 0.0,
        }


class MomentumStrategy:
    """
    Simple momentum strategy based on recent price movements.
    """

    def __init__(self, lookback_window: int = 10, threshold: float = 0.02,
                 transaction_cost: float = 0.001):
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.name = f"Momentum ({lookback_window}d, {threshold:.1%})"

    def fit(self, train_data: pd.DataFrame) -> None:
        pass 

    def get_name(self) -> str:
        return self.name

    def calculate_momentum(self, prices: List[float]) -> List[float]:
        momentum = [0.0] * len(prices)
        for i in range(self.lookback_window, len(prices)):
            if prices[i - self.lookback_window] > 0:
                past_price = prices[i - self.lookback_window]
                current_price = prices[i]
                momentum[i] = (current_price - past_price) / past_price
        return momentum

    def predict_signals(self, data: pd.DataFrame) -> List[str]:
        """
        Generate trading signals based on momentum.
        Args:
            data (pd.DataFrame): Price data, expects 'adjusted_close' or 'close' column.
        Returns:
            List[str]: List of signals ('buy', 'sell', 'hold').
        """
        if 'adjusted_close' in data.columns:
            prices = data['adjusted_close'].tolist()
        elif 'close' in data.columns:
            prices = data['close'].tolist()
        else:
            logger.error("Price data ('adjusted_close' or 'close') not found for Momentum signals.")
            return ['hold'] * len(data)

        if len(prices) < self.lookback_window:
            return ['hold'] * len(prices)

        momentum = self.calculate_momentum(prices)
        signals = ['hold'] * len(prices)
        position = 'out'
        for i, mom in enumerate(momentum):
            if mom > self.threshold and position == 'out':
                signals[i] = 'buy'
                position = 'in'
            elif mom < -self.threshold and position == 'in':
                signals[i] = 'sell'
                position = 'out'
            elif mom < 0 and position == 'in': 
                signals[i] = 'sell'
                position = 'out'
        return signals

    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        if len(prices) < self.lookback_window + 1:
            return self._empty_result(initial_capital)

        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected in Momentum strategy")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < self.lookback_window + 1:
                return self._empty_result(initial_capital)
            prices = prices_array[valid_indices].tolist()
            if len(prices) < self.lookback_window + 1:
                return self._empty_result(initial_capital)


        signals = self.predict_signals(pd.DataFrame({'close': prices}))

        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades_raw = []
        position = 'out'

        for day, (price_point, signal) in enumerate(zip(prices, signals)):
            if signal == 'buy' and position == 'out' and cash > 0:
                trade_cost = price_point * (1 + self.transaction_cost)
                if trade_cost == 0:
                    shares_to_buy = 0
                else:
                    shares_to_buy = cash / trade_cost
                cost = shares_to_buy * trade_cost
                if cost <= cash and shares_to_buy > 0:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'in'
                    trades_raw.append({
                        'action': 'buy', 'day': day, 'price': price_point,
                        'shares': shares_to_buy, 'cost': cost
                    })
            elif signal == 'sell' and position == 'in' and shares > 0:
                proceeds = shares * price_point * (1 - self.transaction_cost)
                cash += proceeds
                trades_raw.append({
                    'action': 'sell', 'day': day, 'price': price_point,
                    'shares': shares, 'proceeds': proceeds
                })
                shares = 0
                position = 'out'
            
            portfolio_value = cash + shares * price_point
            portfolio_values.append(portfolio_value)

        if shares > 0 and prices:
            final_proceeds = shares * prices[-1] * (1 - self.transaction_cost)
            cash += final_proceeds
            trades_raw.append({
                'action': 'sell', 'day': len(prices) - 1, 'price': prices[-1],
                'shares': shares, 'proceeds': final_proceeds
            })

        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        processed_trades = self._process_trades(trades_raw)

        if PerformanceAnalyzer is not None:
            try:
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.comprehensive_analysis(
                    portfolio_values=portfolio_values,
                    trades=processed_trades
                )
            except Exception as e:
                logger.warning(f"PerformanceAnalyzer failed for Momentum: {e}, using fallback")
                metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)
        else:
            metrics = self._calculate_basic_metrics(portfolio_values, processed_trades, initial_capital)

        return {
            'strategy': self.name, 'total_return': total_return, 'final_value': final_value,
            'portfolio_values': portfolio_values, 'trades': processed_trades,
            'num_transactions': len(processed_trades), 'max_profit': final_value - initial_capital,
            'signals': signals, 'metrics': metrics
        }

    def _process_trades(self, raw_trades: List[Dict]) -> List[Dict]:
        processed = []
        buy_trade = None
        for trade in raw_trades:
            if trade['action'] == 'buy':
                buy_trade = trade
            elif trade['action'] == 'sell' and buy_trade is not None:
                profit = trade['proceeds'] - buy_trade['cost']
                processed.append({
                    'buy_day': buy_trade['day'], 'buy_price': buy_trade['price'],
                    'sell_day': trade['day'], 'sell_price': trade['price'],
                    'profit': profit, 'shares': buy_trade['shares'], # Use shares from buy_trade
                    'duration': trade['day'] - buy_trade['day']
                })
                buy_trade = None
        return processed

    def _calculate_basic_metrics(self, portfolio_values: List[float],
                                trades: List[Dict], initial_capital: float) -> Dict:
        if not portfolio_values or initial_capital <= 0:
             return self._empty_metrics_structure(initial_capital if initial_capital > 0 else 0.0)

        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1]) if len(portfolio_values) > 1 else np.array([])
        returns = returns[np.isfinite(returns)]

        win_rate_val = 0.0
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            if len(trades) > 0:
                win_rate_val = profitable_trades / len(trades)

        total_return_val = (portfolio_values[-1] - initial_capital) / initial_capital
        volatility_val = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        sharpe_ratio_val = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            daily_rf_rate = 0.02 / 252
            excess_returns = returns - daily_rf_rate
            sharpe_ratio_val = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

        metrics = {
            'total_return': total_return_val,
            'annualized_return': ((portfolio_values[-1] / initial_capital) ** (252 / len(portfolio_values)) - 1) if len(portfolio_values) > 1 else 0.0,
            'volatility': volatility_val if np.isfinite(volatility_val) else 0.0,
            'sharpe_ratio': sharpe_ratio_val if np.isfinite(sharpe_ratio_val) else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'number_of_trades': len(trades),
            'win_rate': win_rate_val,
            'initial_value': initial_capital,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'drawdown_duration': 0,
            'recovery_time': 0, 'value_at_risk_95': 0.0, 'conditional_var_95': 0.0,
        }
        return metrics

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        if not portfolio_values: return 0.0
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        peak[peak == 0] = 1e-8
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0
        
    def _empty_result(self, initial_capital: float) -> Dict:
        return {
            'strategy': self.name, 'total_return': 0.0, 'final_value': initial_capital,
            'portfolio_values': [initial_capital] if initial_capital > 0 else [0.0],
            'trades': [], 'num_transactions': 0, 'max_profit': 0.0, 'signals': [],
            'metrics': self._empty_metrics_structure(initial_capital)
        }

    def _empty_metrics_structure(self, initial_capital = 0.0) -> Dict:
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'number_of_trades': 0,
            'win_rate': 0.0, 'initial_value': initial_capital, 'final_value': initial_capital,
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'drawdown_duration': 0,
            'recovery_time': 0, 'value_at_risk_95': 0.0, 'conditional_var_95': 0.0,
        }


class StrategyComparator:
    """
    Compare multiple trading strategies on the same data.
    FIXED: Enhanced comparison with better result formatting and validation.
    """

    def __init__(self, strategies: List, initial_capital: float = 100000):
        self.strategies = strategies
        self.initial_capital = initial_capital

    def compare(self, prices: pd.Series, dates: Optional[pd.DatetimeIndex] = None) -> Tuple[pd.DataFrame, Dict]:
        results = []
        individual_results = {}
        price_list = prices.tolist() if hasattr(prices, 'tolist') else list(prices)

        for strategy in self.strategies:
            try:
                result = strategy.execute(price_list, self.initial_capital)
                metrics = result.get('metrics', {})
                portfolio_values = result.get('portfolio_values', [])
                
                returns = np.array([])
                if portfolio_values and len(portfolio_values) > 1:
                    pv_array = np.array(portfolio_values)
                    # Pastikan tidak ada pembagian dengan nol atau nilai tak terhingga
                    pv_array_shifted = np.roll(pv_array, 1)
                    pv_array_shifted[0] = pv_array[0] # Hindari nilai tak terdefinisi pada elemen pertama
                    valid_denominator = pv_array_shifted != 0
                    
                    temp_returns = np.full_like(pv_array, np.nan, dtype=np.double)
                    np.divide(np.diff(pv_array), pv_array_shifted[1:][valid_denominator[1:]], out=temp_returns[1:][valid_denominator[1:]], where=valid_denominator[1:])
                    returns = temp_returns[1:][np.isfinite(temp_returns[1:])]


                strategy_metrics = {
                    'Strategy': strategy.get_name(),
                    'Total_Return': metrics.get('total_return', 0.0),
                    'Final_Value': metrics.get('final_value', self.initial_capital),
                    'Max_Profit': result.get('max_profit', 0.0),
                    'Num_Transactions': metrics.get('number_of_trades', 0),
                    'Annualized_Return': metrics.get('annualized_return', 0.0),
                    'Volatility': metrics.get('volatility', 0.0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0.0),
                    'Max_Drawdown': metrics.get('max_drawdown', 0.0),
                    'Win_Rate': metrics.get('win_rate', 0.0)
                }
                results.append(strategy_metrics)
                individual_results[strategy.get_name()] = result
            except Exception as e:
                logger.error(f"Error executing {strategy.get_name()}: {e}")
                error_metrics = {
                    'Strategy': strategy.get_name(), 'Total_Return': 0.0, 'Final_Value': self.initial_capital,
                    'Max_Profit': 0.0, 'Num_Transactions': 0, 'Annualized_Return': 0.0,
                    'Volatility': 0.0, 'Sharpe_Ratio': 0.0, 'Max_Drawdown': 0.0, 'Win_Rate': 0.0
                }
                results.append(error_metrics)
                individual_results[strategy.get_name()] = {'error': str(e)}
        return pd.DataFrame(results), individual_results

    def _calculate_annualized_return(self, portfolio_values: List[float]) -> float:
        if len(portfolio_values) < 2 or portfolio_values[0] == 0: return 0.0
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        years = len(portfolio_values) / 252
        if years <= 0: return 0.0
        try:
            annualized = ((1 + total_return) ** (1 / years)) - 1
            return annualized if np.isfinite(annualized) else 0.0
        except: return 0.0

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or np.std(returns) == 0: return 0.0
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        return sharpe if np.isfinite(sharpe) else 0.0

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        if not portfolio_values: return 0.0
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        peak[peak == 0] = 1e-8
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        if not trades: return 0.0
        profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return profitable_trades / len(trades) if len(trades) > 0 else 0.0