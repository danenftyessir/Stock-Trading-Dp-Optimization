"""
Dynamic Programming Implementation for Stock Trading Optimization.
Implements optimal k-transaction profit maximization algorithm.
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
    Implements the classic DP solution with O(n*k) time complexity.
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
        
        # If k >= n//2, we can make unlimited transactions
        if k >= n // 2:
            return self._unlimited_transactions(prices)
        
        # DP table: dp[t][i] = maximum profit with at most t transactions by day i
        # Use buy[t][i] and sell[t][i] to track buy/sell states
        buy = [[-float('inf')] * n for _ in range(k + 1)]
        sell = [[0] * n for _ in range(k + 1)]
        
        # Initialize first day
        for t in range(k + 1):
            buy[t][0] = -prices[0] * (1 + self.transaction_cost)
            sell[t][0] = 0
        
        # Fill DP table
        for i in range(1, n):
            for t in range(k + 1):
                # Sell state: either sell today or hold previous sell state
                if t > 0:
                    sell_today = buy[t-1][i-1] + prices[i] * (1 - self.transaction_cost)
                    sell[t][i] = max(sell[t][i-1], sell_today)
                else:
                    sell[t][i] = sell[t][i-1]
                
                # Buy state: either buy today or hold previous buy state
                buy_today = sell[t][i-1] - prices[i] * (1 + self.transaction_cost)
                buy[t][i] = max(buy[t][i-1], buy_today)
        
        # Find maximum profit
        max_profit = sell[k][n-1]
        max_profit_pct = max_profit / (prices[0] * (1 + self.transaction_cost)) if prices[0] > 0 else 0
        
        # Reconstruct optimal trades
        trades = self._reconstruct_trades(prices, buy, sell, k)
        
        # Cap profit to reasonable range
        max_profit_pct = max(-0.99, min(10.0, max_profit_pct))
        
        return max_profit_pct, trades
    
    def _unlimited_transactions(self, prices: List[float]) -> Tuple[float, List[Dict]]:
        """Handle unlimited transactions case (k >= n//2)."""
        total_profit = 0
        trades = []
        
        i = 0
        while i < len(prices) - 1:
            # Find local minimum
            while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
                i += 1
            
            if i == len(prices) - 1:
                break
                
            buy_day = i
            buy_price = prices[i]
            
            # Find local maximum
            while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
                i += 1
            
            sell_day = i
            sell_price = prices[i]
            
            # Calculate profit with transaction costs
            buy_cost = buy_price * (1 + self.transaction_cost)
            sell_proceeds = sell_price * (1 - self.transaction_cost)
            profit = sell_proceeds - buy_cost
            
            if profit > 0:
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
        
        # Cap to reasonable range
        profit_pct = max(-0.99, min(10.0, profit_pct))
        
        return profit_pct, trades
    
    def _reconstruct_trades(self, prices: List[float], buy: List[List[float]], 
                           sell: List[List[float]], k: int) -> List[Dict]:
        """Reconstruct optimal trades from DP table."""
        trades = []
        i = len(prices) - 1
        t = k
        
        while i > 0 and t > 0:
            # Check if we sold at day i
            if t > 0 and sell[t][i] != sell[t][i-1]:
                # We sold at day i, find corresponding buy
                sell_day = i
                sell_price = prices[i]
                
                # Find the buy day for this transaction
                j = i - 1
                while j >= 0 and buy[t-1][j] == buy[t-1][j-1] if j > 0 else False:
                    j -= 1
                
                if j >= 0:
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
                    
                    i = j - 1
                    t -= 1
                else:
                    i -= 1
            else:
                i -= 1
        
        # Reverse to get chronological order
        trades.reverse()
        return trades
    
    def backtest(self, prices: List[float], dates: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive backtesting of the DP strategy.
        
        Args:
            prices (List[float]): Historical stock prices
            dates (Optional[List[str]]): Date labels
            
        Returns:
            Dict: Comprehensive backtesting results
        """
        logger.info(f"Running backtest with {len(prices)} price points")
        
        # Get optimal strategy
        max_profit_pct, trades = self.optimize_profit(prices)
        
        # Simulate portfolio performance
        portfolio_values = self._simulate_portfolio(prices, trades)
        
        # Calculate performance metrics
        from ..analysis.performance_metrics import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        
        metrics = analyzer.comprehensive_analysis(
            portfolio_values=portfolio_values,
            trades=trades,
            dates=dates
        )
        
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
    
    def _simulate_portfolio(self, prices: List[float], trades: List[Dict], 
                           initial_capital: float = 100000) -> List[float]:
        """Simulate portfolio performance given optimal trades."""
        portfolio_values = []
        cash = initial_capital
        shares = 0
        
        # Create trade schedule
        trade_schedule = {}
        for trade in trades:
            buy_day = trade['buy_day']
            sell_day = trade['sell_day']
            
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
                    if action == 'buy' and cash > 0:
                        # Buy shares
                        shares_to_buy = cash / (price * (1 + self.transaction_cost))
                        cost = shares_to_buy * price * (1 + self.transaction_cost)
                        
                        if cost <= cash:
                            shares += shares_to_buy
                            cash -= cost
                    
                    elif action == 'sell' and shares > 0:
                        # Sell shares
                        proceeds = shares * price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values


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
        
        for symbol, prices in stock_data.items():
            logger.info(f"Optimizing {symbol}")
            
            # Create trader for this stock
            trader = DynamicProgrammingTrader(
                max_transactions=self.max_transactions,
                transaction_cost=self.transaction_cost
            )
            
            # Optimize individual stock
            max_profit, trades = trader.optimize_profit(prices)
            
            # Store results
            results[symbol] = {
                'max_profit': max_profit,
                'trades': trades,
                'trader': trader
            }
            
            total_profits.append(max_profit)
            self.traders[symbol] = trader
        
        # Calculate portfolio-level metrics
        results['total_portfolio_return'] = np.mean(total_profits)
        results['portfolio_std'] = np.std(total_profits)
        results['num_stocks'] = len(stock_data)
        
        logger.info(f"Portfolio optimization completed. Average return: {results['total_portfolio_return']:.2%}")
        
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
                trader = self.traders[symbol]
                backtest_result = trader.backtest(prices, dates)
                portfolio_results[symbol] = backtest_result
        
        return portfolio_results