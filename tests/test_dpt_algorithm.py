"""
Unit Tests for Dynamic Programming Trading Algorithm.
Tests the corrected DP implementation for realistic performance.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.dynamic_programing import DynamicProgrammingTrader, DPPortfolioOptimizer


class TestDynamicProgrammingTrader:
    """Test the corrected Dynamic Programming trader implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create realistic test price data
        np.random.seed(42)
        base_price = 100
        num_days = 100
        
        # Generate realistic price series with moderate volatility
        returns = np.random.normal(0.0008, 0.02, num_days)  # ~20% annual volatility
        prices = [base_price]
        
        for i in range(num_days - 1):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1.0))  # Ensure price stays positive
        
        self.test_prices = prices
        self.test_dates = [(datetime.now() - timedelta(days=num_days-i)).strftime('%Y-%m-%d') 
                          for i in range(num_days)]
    
    def test_trader_initialization(self):
        """Test trader initialization."""
        trader = DynamicProgrammingTrader(max_transactions=5, transaction_cost=0.001)
        
        assert trader.max_transactions == 5
        assert trader.transaction_cost == 0.001
        assert trader.trades_history == []
    
    def test_optimize_profit_basic(self):
        """Test basic profit optimization."""
        trader = DynamicProgrammingTrader(max_transactions=2, transaction_cost=0.0)
        
        # Simple test case: [100, 50, 150, 75, 200]
        simple_prices = [100, 50, 150, 75, 200]
        
        max_profit_pct, trades = trader.optimize_profit(simple_prices)
        
        # Check that profit is reasonable (should be positive but not extreme)
        assert max_profit_pct >= 0
        assert max_profit_pct <= 2.0  # Max 200% return for this simple case
        assert len(trades) <= 2  # Should not exceed max_transactions
        
        # Verify trades make sense
        for trade in trades:
            assert 'buy_day' in trade
            assert 'sell_day' in trade
            assert 'buy_price' in trade
            assert 'sell_price' in trade
            assert trade['sell_day'] > trade['buy_day']
    
    def test_optimize_profit_realistic_data(self):
        """Test optimization with realistic market data."""
        trader = DynamicProgrammingTrader(max_transactions=5, transaction_cost=0.001)
        
        max_profit_pct, trades = trader.optimize_profit(self.test_prices)
        
        # Check that returns are realistic (not >1000%)
        assert max_profit_pct >= -0.9  # Max loss 90%
        assert max_profit_pct <= 5.0   # Max gain 500% (still high but more realistic)
        
        # Check number of trades
        assert len(trades) <= 5
        assert len(trades) >= 0
        
        # Verify trade sequence
        for i, trade in enumerate(trades):
            assert trade['buy_day'] < trade['sell_day']
            if i > 0:
                # Next buy should be after previous sell
                assert trade['buy_day'] >= trades[i-1]['sell_day']
    
    def test_transaction_cost_impact(self):
        """Test that transaction costs reduce profits appropriately."""
        # Test with same data but different transaction costs
        trader_no_cost = DynamicProgrammingTrader(max_transactions=3, transaction_cost=0.0)
        trader_with_cost = DynamicProgrammingTrader(max_transactions=3, transaction_cost=0.01)
        
        profit_no_cost, _ = trader_no_cost.optimize_profit(self.test_prices)
        profit_with_cost, _ = trader_with_cost.optimize_profit(self.test_prices)
        
        # Profit with transaction costs should be less than without
        assert profit_with_cost <= profit_no_cost
    
    def test_k_parameter_impact(self):
        """Test that higher k values generally allow for higher profits."""
        profits = []
        
        for k in [1, 2, 5, 10]:
            trader = DynamicProgrammingTrader(max_transactions=k, transaction_cost=0.001)
            profit, trades = trader.optimize_profit(self.test_prices)
            profits.append(profit)
            
            # Number of trades should not exceed k
            assert len(trades) <= k
        
        # Generally, higher k should allow for same or better performance
        # (Though not always strictly increasing due to transaction costs)
        assert profits[-1] >= profits[0]  # k=10 should be >= k=1
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        trader = DynamicProgrammingTrader(max_transactions=2, transaction_cost=0.001)
        
        # Empty prices
        profit, trades = trader.optimize_profit([])
        assert profit == 0.0
        assert trades == []
        
        # Single price
        profit, trades = trader.optimize_profit([100])
        assert profit == 0.0
        assert trades == []
        
        # Two identical prices
        profit, trades = trader.optimize_profit([100, 100])
        assert profit <= 0.001  # Should be minimal due to transaction costs
        
        # Declining prices
        declining_prices = [100, 90, 80, 70, 60]
        profit, trades = trader.optimize_profit(declining_prices)
        assert profit >= -0.5  # Should not lose more than 50%
    
    def test_backtest_functionality(self):
        """Test backtesting functionality."""
        trader = DynamicProgrammingTrader(max_transactions=3, transaction_cost=0.001)
        
        backtest_results = trader.backtest(self.test_prices, self.test_dates)
        
        # Check result structure
        assert 'max_profit' in backtest_results
        assert 'trades' in backtest_results
        assert 'portfolio_values' in backtest_results
        assert 'metrics' in backtest_results
        
        # Check metrics are realistic
        metrics = backtest_results['metrics']
        
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['volatility'], float)
        assert isinstance(metrics['max_drawdown'], float)
        
        # Check realistic bounds
        assert -0.99 <= metrics['total_return'] <= 10.0  # Between -99% and 1000%
        assert -5.0 <= metrics['sharpe_ratio'] <= 5.0    # Realistic Sharpe ratio range
        assert 0.0 <= metrics['volatility'] <= 5.0       # Reasonable volatility
        assert 0.0 <= metrics['max_drawdown'] <= 1.0     # Drawdown between 0-100%
        
        # Portfolio values should be realistic
        portfolio_values = backtest_results['portfolio_values']
        assert len(portfolio_values) == len(self.test_prices)
        assert all(v > 0 for v in portfolio_values)  # All values should be positive
    
    def test_unlimited_transactions_case(self):
        """Test the unlimited transactions case (when 2k >= n)."""
        # Small price array with large k
        small_prices = [100, 120, 90, 130, 110]
        trader = DynamicProgrammingTrader(max_transactions=10, transaction_cost=0.001)  # 2k >= n
        
        profit, trades = trader.optimize_profit(small_prices)
        
        # Should handle unlimited case properly
        assert profit >= 0
        assert profit <= 1.0  # Should not be extreme
        assert len(trades) >= 0
    
    def test_performance_metrics_accuracy(self):
        """Test that performance metrics are calculated accurately."""
        trader = DynamicProgrammingTrader(max_transactions=2, transaction_cost=0.001)
        
        # Known test case
        test_prices = [100, 110, 90, 120, 95, 130]
        
        backtest_results = trader.backtest(test_prices)
        metrics = backtest_results['metrics']
        
        # Basic sanity checks
        portfolio_values = backtest_results['portfolio_values']
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        expected_total_return = (final_value - initial_value) / initial_value
        
        # Total return should match calculation
        assert abs(metrics['total_return'] - expected_total_return) < 0.001
        
        # Sharpe ratio should be finite and reasonable
        assert np.isfinite(metrics['sharpe_ratio'])
        assert -5.0 <= metrics['sharpe_ratio'] <= 5.0


class TestDPPortfolioOptimizer:
    """Test the portfolio-level optimizer."""
    
    def setup_method(self):
        """Setup test data for portfolio optimization."""
        np.random.seed(42)
        
        # Create test data for multiple stocks
        self.portfolio_data = {}
        
        for symbol in ['STOCK1', 'STOCK2', 'STOCK3']:
            # Generate different price patterns for each stock
            base_price = np.random.uniform(50, 150)
            returns = np.random.normal(0.0005, 0.015, 50)  # 50 days of data
            
            prices = [base_price]
            for ret in returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))
            
            self.portfolio_data[symbol] = prices
    
    def test_portfolio_optimizer_initialization(self):
        """Test portfolio optimizer initialization."""
        optimizer = DPPortfolioOptimizer(max_transactions=3, transaction_cost=0.001)
        
        assert optimizer.max_transactions == 3
        assert optimizer.transaction_cost == 0.001
        assert optimizer.traders == {}
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization across multiple stocks."""
        optimizer = DPPortfolioOptimizer(max_transactions=2, transaction_cost=0.001)
        
        results = optimizer.optimize_portfolio(self.portfolio_data)
        
        # Check result structure
        assert 'total_portfolio_return' in results
        
        # Check individual stock results
        for symbol in self.portfolio_data.keys():
            assert symbol in results
            assert 'max_profit' in results[symbol]
            assert 'trades' in results[symbol]
            assert 'trader' in results[symbol]
            
            # Check that profits are realistic
            profit = results[symbol]['max_profit']
            assert -0.9 <= profit <= 5.0  # Reasonable profit range
        
        # Total portfolio return should be reasonable
        total_return = results['total_portfolio_return']
        assert -0.9 <= total_return <= 5.0


class TestPerformanceBounds:
    """Test that performance metrics stay within realistic bounds."""
    
    def test_extreme_price_movements(self):
        """Test algorithm behavior with extreme price movements."""
        trader = DynamicProgrammingTrader(max_transactions=5, transaction_cost=0.001)
        
        # Test with extreme bull market
        bull_prices = [100 * (1.01 ** i) for i in range(50)]  # 1% daily growth
        profit, trades = trader.optimize_profit(bull_prices)
        
        # Should be profitable but not unrealistic
        assert profit > 0
        assert profit <= 10.0  # Max 1000% return
        
        # Test with extreme bear market
        bear_prices = [100 * (0.99 ** i) for i in range(50)]  # 1% daily decline
        profit, trades = trader.optimize_profit(bear_prices)
        
        # Should limit losses
        assert profit >= -0.99  # Max 99% loss
    
    def test_sharpe_ratio_bounds(self):
        """Test that Sharpe ratios stay within realistic bounds."""
        trader = DynamicProgrammingTrader(max_transactions=3, transaction_cost=0.001)
        
        # Test multiple scenarios
        scenarios = [
            [100, 105, 95, 110, 90, 115],  # Volatile but trending up
            [100, 101, 102, 103, 104, 105],  # Steady growth
            [100, 100, 100, 100, 100, 100],  # No movement
            [100, 95, 105, 90, 110, 85]   # High volatility
        ]
        
        for prices in scenarios:
            backtest_results = trader.backtest(prices)
            sharpe_ratio = backtest_results['metrics']['sharpe_ratio']
            
            # Sharpe ratio should be within realistic bounds
            assert -5.0 <= sharpe_ratio <= 5.0
            assert np.isfinite(sharpe_ratio)
    
    def test_return_consistency(self):
        """Test that returns are consistent across different calculation methods."""
        trader = DynamicProgrammingTrader(max_transactions=2, transaction_cost=0.001)
        
        test_prices = [100, 120, 90, 130, 110]
        
        # Get profit from optimization
        max_profit_pct, trades = trader.optimize_profit(test_prices)
        
        # Get metrics from backtesting
        backtest_results = trader.backtest(test_prices)
        total_return = backtest_results['metrics']['total_return']
        
        # They should be reasonably close (allowing for some calculation differences)
        assert abs(max_profit_pct - total_return) < 0.1  # Within 10 percentage points


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])