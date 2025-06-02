"""
Baseline Trading Strategies for comparison with Dynamic Programming approach.
Implements Buy & Hold, Moving Average Crossover, and other simple strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
        self.name = "Buy & Hold"
        
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
        
        trades = [{
            'buy_day': 0,
            'buy_price': buy_price,
            'sell_day': len(prices) - 1,
            'sell_price': sell_price,
            'profit': final_value - initial_capital,
            'shares': shares
        }]
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'num_transactions': 1,
            'max_profit': final_value - initial_capital
        }
    
    def _empty_result(self, initial_capital: float) -> Dict:
        """Return empty result for invalid inputs."""
        return {
            'strategy': self.name,
            'total_return': 0.0,
            'final_value': initial_capital,
            'portfolio_values': [initial_capital],
            'trades': [],
            'num_transactions': 0,
            'max_profit': 0.0
        }


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover strategy.
    Buy when short MA crosses above long MA, sell when it crosses below.
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
        
        Returns:
            List[str]: Signal for each day ('buy', 'sell', 'hold')
        """
        short_ma, long_ma = self.calculate_moving_averages(prices)
        signals = ['hold'] * len(prices)
        
        for i in range(1, len(prices)):
            if (not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]) and
                not np.isnan(short_ma[i-1]) and not np.isnan(long_ma[i-1])):
                
                # Buy signal: short MA crosses above long MA
                if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                    signals[i] = 'buy'
                # Sell signal: short MA crosses below long MA
                elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                    signals[i] = 'sell'
                    
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
            
        signals = self.generate_signals(prices)
        
        # Simulate trading
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 'out'  # 'in' or 'out' of market
        
        for day, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 'buy' and position == 'out' and cash > 0:
                # Buy shares
                shares_to_buy = cash / (price * (1 + self.transaction_cost))
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                if cost <= cash:
                    shares += shares_to_buy
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
        
        final_value = cash + shares * prices[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Process trades into buy-sell pairs
        processed_trades = self._process_trades(trades)
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': processed_trades,
            'num_transactions': len(processed_trades),
            'max_profit': final_value - initial_capital,
            'signals': signals
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
                    'shares': trade['shares']
                })
                buy_trade = None
                
        return processed
    
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
            'signals': []
        }


class MomentumStrategy:
    """
    Simple momentum strategy based on recent price movements.
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
        """Calculate momentum indicator."""
        momentum = [0.0] * len(prices)
        
        for i in range(self.lookback_window, len(prices)):
            past_price = prices[i - self.lookback_window]
            current_price = prices[i]
            momentum[i] = (current_price - past_price) / past_price
            
        return momentum
    
    def generate_signals(self, prices: List[float]) -> List[str]:
        """Generate trading signals based on momentum."""
        momentum = self.calculate_momentum(prices)
        signals = ['hold'] * len(prices)
        
        for i, mom in enumerate(momentum):
            if mom > self.threshold:
                signals[i] = 'buy'
            elif mom < -self.threshold:
                signals[i] = 'sell'
                
        return signals
    
    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """Execute momentum strategy."""
        if len(prices) < self.lookback_window + 1:
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
                shares_to_buy = cash / (price * (1 + self.transaction_cost))
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                if cost <= cash:
                    shares += shares_to_buy
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
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': processed_trades,
            'num_transactions': len(processed_trades),
            'max_profit': final_value - initial_capital,
            'signals': signals
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
                    'shares': trade['shares']
                })
                buy_trade = None
                
        return processed
    
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
            'signals': []
        }


class RandomStrategy:
    """
    Random trading strategy for baseline comparison.
    Makes random buy/sell decisions with given probability.
    """
    
    def __init__(self, trade_probability: float = 0.1, 
                 transaction_cost: float = 0.001, random_seed: int = 42):
        """
        Initialize Random strategy.
        
        Args:
            trade_probability (float): Probability of making a trade each day
            transaction_cost (float): Transaction cost as percentage
            random_seed (int): Random seed for reproducibility
        """
        self.trade_probability = trade_probability
        self.transaction_cost = transaction_cost
        self.random_seed = random_seed
        self.name = f"Random ({trade_probability:.1%})"
        
    def execute(self, prices: List[float], initial_capital: float = 100000) -> Dict:
        """Execute random trading strategy."""
        np.random.seed(self.random_seed)
        
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 'out'
        
        for day, price in enumerate(prices):
            # Random decision to trade
            if np.random.random() < self.trade_probability:
                if position == 'out' and cash > 0:
                    # Random buy
                    shares_to_buy = cash / (price * (1 + self.transaction_cost))
                    cost = shares_to_buy * price * (1 + self.transaction_cost)
                    
                    shares += shares_to_buy
                    cash -= cost
                    position = 'in'
                    
                    trades.append({
                        'action': 'buy',
                        'day': day,
                        'price': price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
                    
                elif position == 'in' and shares > 0:
                    # Random sell
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
            
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        processed_trades = self._process_trades(trades)
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trades': processed_trades,
            'num_transactions': len(processed_trades),
            'max_profit': final_value - initial_capital
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
                    'shares': trade['shares']
                })
                buy_trade = None
                
        return processed


class StrategyComparator:
    """
    Compare multiple trading strategies on the same data.
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
        
    def compare(self, prices: List[float]) -> pd.DataFrame:
        """
        Compare all strategies on given price data.
        
        Args:
            prices (List[float]): Historical stock prices
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for strategy in self.strategies:
            try:
                result = strategy.execute(prices, self.initial_capital)
                
                # Calculate additional metrics
                portfolio_values = result['portfolio_values']
                returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
                
                metrics = {
                    'Strategy': result['strategy'],
                    'Total Return': result['total_return'],
                    'Final Value': result['final_value'],
                    'Max Profit': result['max_profit'],
                    'Num Transactions': result['num_transactions'],
                    'Annualized Return': ((result['final_value'] / self.initial_capital) ** 
                                        (252 / len(prices))) - 1,
                    'Volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                    'Sharpe Ratio': (np.mean(returns) / np.std(returns) * np.sqrt(252) 
                                   if len(returns) > 0 and np.std(returns) > 0 else 0),
                    'Max Drawdown': self._calculate_max_drawdown(portfolio_values)
                }
                
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error executing {strategy.name}: {e}")
                
        return pd.DataFrame(results)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd