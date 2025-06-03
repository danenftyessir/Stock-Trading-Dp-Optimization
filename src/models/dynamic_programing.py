"""
Dynamic Programming Implementation for Stock Trading Optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DynamicProgrammingTrader:
    """
    Dynamic Programming trader for optimal k-transaction profit maximization.
    """
    
    def __init__(self, max_transactions: int, transaction_cost: float = 0.001):
        """
        Initialize DP trader.
        
        Args:
            max_transactions (int): Maximum number of transactions (k)
            transaction_cost (float): Transaction cost as percentage
        """
        self.max_transactions = max_transactions
        self.transaction_cost = transaction_cost
        self.trades_history = []
        
    def optimize_profit(self, prices: List[float]) -> Tuple[float, List[Dict]]:
        """
        Find optimal trading strategy using Dynamic Programming.
        
        Args:
            prices (List[float]): Daily stock prices
            
        Returns:
            Tuple[float, List[Dict]]: Maximum profit percentage and optimal trades
        """
        if not prices or len(prices) < 2:
            return 0.0, []
        
        n = len(prices)
        k = self.max_transactions
        
        # Validate inputs
        prices_array = np.array(prices)
        if np.any(prices_array <= 0):
            logger.warning("Non-positive prices detected, filtering them out")
            valid_indices = prices_array > 0
            if np.sum(valid_indices) < 2:
                return 0.0, []
            prices = prices_array[valid_indices].tolist()
            n = len(prices)
        
        # If k >= n//2, we can make unlimited transactions
        if k >= n // 2:
            return self._unlimited_transactions(prices)
        
        # buy[i][t] = max profit after at most t transactions, currently holding stock on day i
        # sell[i][t] = max profit after at most t transactions, not holding stock on day i
        buy = np.full((n, k + 1), -np.inf)
        sell = np.zeros((n, k + 1))
        
        # Initialize first day
        for t in range(k + 1):
            buy[0][t] = -prices[0] * (1 + self.transaction_cost)
            sell[0][t] = 0
        
        # Fill DP table with corrected transitions
        for i in range(1, n):
            for t in range(k + 1):
                # Sell state: max of (hold previous sell, sell today)
                sell[i][t] = sell[i-1][t]  # Hold previous sell state
                if t > 0 and buy[i-1][t-1] != -np.inf:
                    # Can sell today (completing a transaction)
                    sell_profit = buy[i-1][t-1] + prices[i] * (1 - self.transaction_cost)
                    sell[i][t] = max(sell[i][t], sell_profit)
                
                # Buy state: max of (hold previous buy, buy today)
                buy[i][t] = buy[i-1][t]  # Hold previous buy state
                if sell[i-1][t] > -np.inf:
                    # Can buy today
                    buy_cost = sell[i-1][t] - prices[i] * (1 + self.transaction_cost)
                    buy[i][t] = max(buy[i][t], buy_cost)
        
        # Find maximum profit
        max_profit = sell[n-1][k]
        
        # Calculate profit percentage relative to initial investment
        initial_investment = prices[0] * (1 + self.transaction_cost)
        max_profit_pct = max_profit / initial_investment if initial_investment > 0 else 0
        
        # Reconstruct optimal trades
        trades = self._reconstruct_trades_corrected(prices, buy, sell, k, n)
        
        # Validate the result
        if max_profit_pct < -0.99:  # More than 99% loss is unlikely to be optimal
            logger.warning(f"Extreme loss detected: {max_profit_pct:.4f}. Checking calculation...")
            # Return simple buy-and-hold as fallback
            return self._simple_buy_hold(prices)
        
        return max_profit_pct, trades
    
    def _unlimited_transactions(self, prices: List[float]) -> Tuple[float, List[Dict]]:
        """
        Handle unlimited transactions case (k >= n//2).
        """
        total_profit = 0
        trades = []
        
        i = 0
        while i < len(prices) - 1:
            # Find local minimum (buy point)
            while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
                i += 1
            
            if i == len(prices) - 1:
                break
                
            buy_day = i
            buy_price = prices[i]
            
            # Find local maximum (sell point)
            while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
                i += 1
            
            sell_day = i
            sell_price = prices[i]
            
            # Calculate profit with transaction costs
            buy_cost = buy_price * (1 + self.transaction_cost)
            sell_proceeds = sell_price * (1 - self.transaction_cost)
            profit = sell_proceeds - buy_cost
            
            # Only execute trade if it's profitable after costs
            if profit > buy_cost * 0.001:  # At least 0.1% profit after costs
                total_profit += profit
                trades.append({
                    'buy_day': buy_day,
                    'sell_day': sell_day,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit': profit
                })
        
        # Calculate profit percentage
        initial_investment = prices[0] * (1 + self.transaction_cost) if prices[0] > 0 else 1
        profit_pct = total_profit / initial_investment
        
        return profit_pct, trades
    
    def _reconstruct_trades_corrected(self, prices: List[float], buy: np.ndarray, 
                                    sell: np.ndarray, k: int, n: int) -> List[Dict]:
        """
        Reconstruct optimal trades from DP table.
        """
        trades = []
        i = n - 1
        t = k
        holding = False  # Track whether we're currently holding stock
        
        # Work backwards from the end
        while i > 0 and t > 0:
            if not holding:
                # Currently not holding, check if we should have bought at day i
                if sell[i][t] != sell[i-1][t]:
                    # We sold at day i, so we must have been holding
                    holding = True
                    sell_day = i
                    sell_price = prices[i]
                    sell_profit = sell[i][t]
                    
                    # Now find when we bought
                    j = i - 1
                    while j >= 0:
                        if buy[j][t-1] != -np.inf and abs(buy[j][t-1] + prices[i] * (1 - self.transaction_cost) - sell_profit) < 1e-6:
                            buy_day = j
                            buy_price = prices[j]
                            
                            # Calculate profit
                            buy_cost = buy_price * (1 + self.transaction_cost)
                            sell_proceeds = sell_price * (1 - self.transaction_cost)
                            profit = sell_proceeds - buy_cost
                            
                            trades.append({
                                'buy_day': buy_day,
                                'sell_day': sell_day,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'profit': profit
                            })
                            
                            i = j
                            t -= 1
                            holding = False
                            break
                        j -= 1
                    
                    if j < 0:  # Couldn't find matching buy
                        i -= 1
                else:
                    i -= 1
            else:
                # Currently holding, check if we bought at day i
                if buy[i][t] != buy[i-1][t]:
                    # We bought at day i
                    holding = False
                i -= 1
        
        # Reverse to get chronological order
        trades.reverse()
        
        # Validate trades
        validated_trades = []
        for trade in trades:
            if (trade['sell_day'] > trade['buy_day'] and 
                trade['buy_price'] > 0 and 
                trade['sell_price'] > 0):
                validated_trades.append(trade)
        
        return validated_trades
    
    def _simple_buy_hold(self, prices: List[float]) -> Tuple[float, List[Dict]]:
        """
        Simple buy and hold strategy as fallback.
        """
        if len(prices) < 2:
            return 0.0, []
        
        buy_price = prices[0]
        sell_price = prices[-1]
        
        buy_cost = buy_price * (1 + self.transaction_cost)
        sell_proceeds = sell_price * (1 - self.transaction_cost)
        profit = sell_proceeds - buy_cost
        
        profit_pct = profit / buy_cost if buy_cost > 0 else 0
        
        if profit > 0:
            trades = [{
                'buy_day': 0,
                'sell_day': len(prices) - 1,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'profit': profit
            }]
        else:
            trades = []
        
        return profit_pct, trades
    
    def backtest(self, prices: List[float], dates: Optional[List[str]] = None,
                initial_capital: float = 100000) -> Dict:
        """
        Perform comprehensive backtesting of the DP strategy.
        
        Args:
            prices (List[float]): Historical stock prices
            dates (Optional[List[str]]): Date labels
            initial_capital (float): Initial capital
            
        Returns:
            Dict: Comprehensive backtesting results
        """
        logger.info(f"Running backtest with {len(prices)} price points")
        
        # Get optimal strategy
        max_profit_pct, trades = self.optimize_profit(prices)
        
        # Simulate portfolio performance more realistically
        portfolio_values = self._simulate_portfolio_realistic(prices, trades, initial_capital)
        
        # Validate portfolio values
        if len(portfolio_values) != len(prices):
            logger.warning("Portfolio values length mismatch, adjusting...")
            portfolio_values = self._adjust_portfolio_length(portfolio_values, len(prices), initial_capital)
        
        # Calculate performance metrics
        try:
            from ..analysis.performance_metrics import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            
            metrics = analyzer.comprehensive_analysis(
                portfolio_values=portfolio_values,
                trades=trades,
                dates=dates
            )
        except ImportError:
            # Fallback to basic metrics if analyzer not available
            metrics = self._basic_metrics(portfolio_values, trades, initial_capital)
        
        return {
            'max_profit': max_profit_pct,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'metrics': metrics,
            'strategy_params': {
                'max_transactions': self.max_transactions,
                'transaction_cost': self.transaction_cost
            }
        }
    
    def _simulate_portfolio_realistic(self, prices: List[float], trades: List[Dict], 
                                    initial_capital: float = 100000) -> List[float]:
        """
        Simulate portfolio performance with realistic constraints.
        """
        portfolio_values = []
        cash = initial_capital
        shares = 0
        
        # Create trade schedule with validation
        trade_schedule = {}
        for trade in trades:
            buy_day = trade['buy_day']
            sell_day = trade['sell_day']
            
            # Validate trade days
            if buy_day < 0 or sell_day >= len(prices) or buy_day >= sell_day:
                logger.warning(f"Invalid trade detected: buy_day={buy_day}, sell_day={sell_day}")
                continue
            
            if buy_day not in trade_schedule:
                trade_schedule[buy_day] = []
            if sell_day not in trade_schedule:
                trade_schedule[sell_day] = []
            
            trade_schedule[buy_day].append(('buy', trade))
            trade_schedule[sell_day].append(('sell', trade))
        
        # Simulate day by day
        for day, price in enumerate(prices):
            # Execute scheduled trades
            if day in trade_schedule:
                for action, trade in trade_schedule[day]:
                    if action == 'buy' and cash > 0 and shares == 0:  # Only buy if not holding
                        # Buy shares
                        trade_cost = price * (1 + self.transaction_cost)
                        shares_to_buy = cash / trade_cost
                        
                        if shares_to_buy > 0:
                            total_cost = shares_to_buy * trade_cost
                            if total_cost <= cash:
                                shares = shares_to_buy
                                cash -= total_cost
                    
                    elif action == 'sell' and shares > 0:  # Only sell if holding
                        # Sell shares
                        proceeds = shares * price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    def _adjust_portfolio_length(self, portfolio_values: List[float], 
                               target_length: int, initial_capital: float) -> List[float]:
        """Adjust portfolio values list to match target length."""
        if len(portfolio_values) == target_length:
            return portfolio_values
        
        if len(portfolio_values) < target_length:
            # Extend with last value
            last_value = portfolio_values[-1] if portfolio_values else initial_capital
            portfolio_values.extend([last_value] * (target_length - len(portfolio_values)))
        else:
            # Truncate to target length
            portfolio_values = portfolio_values[:target_length]
        
        return portfolio_values
    
    def _basic_metrics(self, portfolio_values: List[float], trades: List[Dict], 
                    initial_capital: float) -> Dict:
        """Calculate basic performance metrics as fallback."""
        if not portfolio_values:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'annualized_return': 0.0,
                'final_value': initial_capital,
                'initial_value': initial_capital,
                'number_of_trades': 0
            }
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        num_trades = len(trades)
        
        # Calculate win rate
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            win_rate = profitable_trades / num_trades
        else:
            win_rate = 0.0
        
        # Calculate returns for risk metrics
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        returns = returns[np.isfinite(returns)]
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        # Calculate Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            # Ensure finite value
            sharpe_ratio = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate annualized return
        if len(portfolio_values) > 1:
            years = len(portfolio_values) / 252
            try:
                annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years)) - 1
                annualized_return = annualized_return if np.isfinite(annualized_return) else 0.0
            except:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_value': portfolio_values[-1],
            'initial_value': initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'annualized_return': annualized_return,
            'number_of_trades': num_trades
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)


class DPPortfolioOptimizer:
    """
    Portfolio-level optimizer using Dynamic Programming for multiple stocks.
    """
    
    def __init__(self, max_transactions: int, transaction_cost: float = 0.001):
        """
        Initialize portfolio optimizer.
        
        Args:
            max_transactions (int): Maximum transactions per stock
            transaction_cost (float): Transaction cost percentage
        """
        self.max_transactions = max_transactions
        self.transaction_cost = transaction_cost
        self.traders = {}
        
    def optimize_portfolio(self, stock_data: Dict[str, List[float]]) -> Dict:
        """
        Optimize trading strategy for a portfolio of stocks.
        
        Args:
            stock_data (Dict[str, List[float]]): Stock symbol -> price list mapping
            
        Returns:
            Dict: Portfolio optimization results
        """
        logger.info(f"Optimizing portfolio with {len(stock_data)} stocks")
        
        results = {}
        total_profits = []
        successful_optimizations = 0
        
        for symbol, prices in stock_data.items():
            logger.info(f"Optimizing {symbol}")
            
            try:
                # Create trader for this stock
                trader = DynamicProgrammingTrader(
                    max_transactions=self.max_transactions,
                    transaction_cost=self.transaction_cost
                )
                
                # Optimize individual stock
                max_profit, trades = trader.optimize_profit(prices)
                
                # Validate results
                if np.isfinite(max_profit) and max_profit > -0.99:
                    # Store results
                    results[symbol] = {
                        'max_profit': max_profit,
                        'trades': trades,
                        'trader': trader,
                        'status': 'success'
                    }
                    
                    total_profits.append(max_profit)
                    successful_optimizations += 1
                    self.traders[symbol] = trader
                else:
                    logger.warning(f"Invalid optimization result for {symbol}: {max_profit}")
                    results[symbol] = {
                        'max_profit': 0.0,
                        'trades': [],
                        'status': 'failed',
                        'error': 'Invalid profit calculation'
                    }
                    
            except Exception as e:
                logger.error(f"Optimization failed for {symbol}: {e}")
                results[symbol] = {
                    'max_profit': 0.0,
                    'trades': [],
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate portfolio-level metrics
        if total_profits:
            results['total_portfolio_return'] = np.mean(total_profits)
            results['portfolio_std'] = np.std(total_profits)
            results['portfolio_sharpe'] = (np.mean(total_profits) / np.std(total_profits)) if np.std(total_profits) > 0 else 0
        else:
            results['total_portfolio_return'] = 0.0
            results['portfolio_std'] = 0.0
            results['portfolio_sharpe'] = 0.0
        
        results['num_stocks'] = len(stock_data)
        results['successful_optimizations'] = successful_optimizations
        results['success_rate'] = successful_optimizations / len(stock_data) if stock_data else 0
        
        logger.info(f"Portfolio optimization completed. Average return: {results['total_portfolio_return']:.2%}, "
                   f"Success rate: {results['success_rate']:.1%}")
        
        return results
    
    def backtest_portfolio(self, stock_data: Dict[str, List[float]], 
                          dates: Optional[List[str]] = None) -> Dict:
        """
        Perform portfolio-level backtesting.
        
        Args:
            stock_data (Dict[str, List[float]]): Stock data
            dates (Optional[List[str]]): Date labels
            
        Returns:
            Dict: Portfolio backtesting results
        """
        portfolio_results = {}
        
        for symbol, prices in stock_data.items():
            if symbol in self.traders:
                try:
                    trader = self.traders[symbol]
                    backtest_result = trader.backtest(prices, dates)
                    portfolio_results[symbol] = backtest_result
                except Exception as e:
                    logger.error(f"Backtesting failed for {symbol}: {e}")
                    portfolio_results[symbol] = {'error': str(e)}
        
        return portfolio_results