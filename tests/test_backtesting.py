"""
Unit Tests for Backtesting Framework.
Tests the comprehensive backtesting system including walk-forward analysis,
out-of-sample testing, and Monte Carlo simulation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.backtesting import (
    BacktestEngine, BacktestConfig, DPTradingStrategy,
    TradingStrategy
)
from models.baseline_strategies import BuyAndHoldStrategy


class MockStrategy(TradingStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, name="MockStrategy", signals=None):
        self.name = name
        self.signals = signals or ['hold'] * 100
        self.fitted = False
    
    def fit(self, train_data: pd.DataFrame) -> None:
        self.fitted = True
    
    def predict_signals(self, data: pd.DataFrame) -> list:
        return self.signals[:len(data)]
    
    def get_name(self) -> str:
        return self.name


class TestBacktestConfig:
    """Test backtesting configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 100000
        assert config.transaction_cost == 0.001
        assert config.train_ratio == 0.7
        assert config.validation_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.walk_forward == True
        assert config.risk_free_rate == 0.02
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            initial_capital=50000,
            transaction_cost=0.005,
            train_ratio=0.8
        )
        
        assert config.initial_capital == 50000
        assert config.transaction_cost == 0.005
        assert config.train_ratio == 0.8


class TestBacktestEngine:
    """Test the main backtesting engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)
        self.engine = BacktestEngine(self.config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        
        self.sample_data = pd.DataFrame({
            'adjusted_close': prices,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.config == self.config
        assert self.engine.performance_analyzer is not None
    
    def test_train_test_split(self):
        """Test data splitting functionality."""
        train_data, test_data = self.engine._train_test_split(self.sample_data)
        
        expected_split = int(len(self.sample_data) * (self.config.train_ratio + self.config.validation_ratio))
        
        assert len(train_data) == expected_split
        assert len(test_data) == len(self.sample_data) - expected_split
        assert len(train_data) + len(test_data) == len(self.sample_data)
    
    def test_train_val_test_split(self):
        """Test three-way data splitting."""
        train_data, val_data, test_data = self.engine._train_val_test_split(self.sample_data)
        
        expected_train = int(len(self.sample_data) * self.config.train_ratio)
        expected_val = int(len(self.sample_data) * self.config.validation_ratio)
        
        assert len(train_data) == expected_train
        assert len(val_data) == expected_val
        assert len(train_data) + len(val_data) + len(test_data) == len(self.sample_data)
    
    def test_execute_trades_simple(self):
        """Test trade execution with simple signals."""
        data = self.sample_data.iloc[:10]
        signals = ['hold', 'buy', 'hold', 'hold', 'sell', 'hold', 'buy', 'hold', 'sell', 'hold']
        
        portfolio_values, trades = self.engine._execute_trades(data, signals, 10000)
        
        assert len(portfolio_values) == len(data)
        assert isinstance(trades, list)
        
        # Should have executed some trades
        assert len(trades) > 0
        
        # Portfolio values should be reasonable
        assert all(val > 0 for val in portfolio_values)
    
    def test_execute_trades_no_signals(self):
        """Test trade execution with no trading signals."""
        data = self.sample_data.iloc[:10]
        signals = ['hold'] * 10
        
        portfolio_values, trades = self.engine._execute_trades(data, signals, 10000)
        
        assert len(portfolio_values) == len(data)
        assert len(trades) == 0  # No trades executed
        
        # Portfolio value should remain constant (just cash)
        assert all(abs(val - 10000) < 1e-6 for val in portfolio_values)
    
    def test_simple_backtest(self):
        """Test simple backtesting functionality."""
        # Create a mock strategy that buys at start and sells at end
        signals = ['buy'] + ['hold'] * 98 + ['sell']
        strategy = MockStrategy(signals=signals)
        
        result = self.engine.simple_backtest(strategy, self.sample_data, "TEST")
        
        # Check result structure
        assert 'strategy' in result
        assert 'symbol' in result
        assert 'backtest_type' in result
        assert 'portfolio_values' in result
        assert 'trades' in result
        assert 'metrics' in result
        
        assert result['strategy'] == strategy.get_name()
        assert result['symbol'] == "TEST"
        assert result['backtest_type'] == 'simple'
        
        # Check that strategy was fitted
        assert strategy.fitted
    
    def test_walk_forward_backtest(self):
        """Test walk-forward backtesting."""
        # Create strategy with simple buy/sell signals
        signals = (['buy'] + ['hold'] * 49 + ['sell'] + ['hold'] * 49) * 10  # Repeat pattern
        strategy = MockStrategy(signals=signals)
        
        # Use smaller window for testing
        self.config.walk_forward_window = 20
        self.config.walk_forward_step = 10
        
        result = self.engine.walk_forward_backtest(strategy, self.sample_data, "TEST")
        
        assert result['backtest_type'] == 'walk_forward'
        assert 'window_size' in result
        assert 'step_size' in result
        assert 'period_results' in result
        
        # Should have multiple periods
        assert len(result['period_results']) > 1
    
    def test_out_of_sample_test(self):
        """Test out-of-sample testing."""
        signals = ['buy'] + ['hold'] * 98 + ['sell']
        strategy = MockStrategy(signals=signals)
        
        result = self.engine.out_of_sample_test(strategy, self.sample_data, "TEST")
        
        assert result['backtest_type'] == 'out_of_sample'
        assert 'validation_results' in result
        assert 'test_results' in result
        
        # Both validation and test should have required components
        for key in ['validation_results', 'test_results']:
            assert 'portfolio_values' in result[key]
            assert 'trades' in result[key]
            assert 'metrics' in result[key]
    
    def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        strategies = [
            MockStrategy("Strategy1", ['buy'] + ['hold'] * 98 + ['sell']),
            MockStrategy("Strategy2", ['hold'] * 50 + ['buy'] + ['hold'] * 48 + ['sell']),
            BuyAndHoldStrategy(transaction_cost=0.0)
        ]
        
        result = self.engine.strategy_comparison(strategies, self.sample_data, "TEST")
        
        assert result['comparison_type'] == 'strategy_comparison'
        assert 'individual_results' in result
        assert 'summary_table' in result
        assert 'best_strategy' in result
        
        # Should have results for all strategies
        assert len(result['individual_results']) == len(strategies)
        
        # Summary table should be a list of dictionaries
        assert isinstance(result['summary_table'], list)
        if result['summary_table']:
            assert isinstance(result['summary_table'][0], dict)
    
    def test_bootstrap_sample(self):
        """Test bootstrap sampling for Monte Carlo."""
        original_data = self.sample_data
        sample_data = self.engine._bootstrap_sample(original_data)
        
        # Sample should have same length as original
        assert len(sample_data) == len(original_data)
        
        # Sample should have same columns
        assert list(sample_data.columns) == list(original_data.columns)
        
        # Index should be reset
        assert sample_data.index.tolist() == list(range(len(sample_data)))


class TestDPTradingStrategy:
    """Test the DP trading strategy wrapper."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = DPTradingStrategy(max_transactions=2, transaction_cost=0.001)
        
        # Sample data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        prices = [100, 105, 95, 110, 108, 120, 115, 125, 118, 130,
                 128, 135, 130, 140, 138, 145, 142, 150, 148, 155]
        
        self.sample_data = pd.DataFrame({
            'adjusted_close': prices,
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.max_transactions == 2
        assert self.strategy.transaction_cost == 0.001
        assert self.strategy.trader is None
    
    def test_fit_method(self):
        """Test strategy fitting."""
        self.strategy.fit(self.sample_data)
        
        assert self.strategy.trader is not None
        assert self.strategy.trader.max_transactions == 2
        assert self.strategy.trader.transaction_cost == 0.001
    
    def test_predict_signals(self):
        """Test signal prediction."""
        self.strategy.fit(self.sample_data)
        signals = self.strategy.predict_signals(self.sample_data)
        
        assert len(signals) == len(self.sample_data)
        assert all(signal in ['buy', 'sell', 'hold'] for signal in signals)
    
    def test_get_name(self):
        """Test strategy name."""
        assert self.strategy.get_name() == "DP_K2"
        
        strategy_k5 = DPTradingStrategy(5)
        assert strategy_k5.get_name() == "DP_K5"


class TestIntegrationScenarios:
    """Integration tests for complete backtesting scenarios."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.config = BacktestConfig(initial_capital=10000, transaction_cost=0.001)
        self.engine = BacktestEngine(self.config)
        
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Generate correlated price series (like real stocks)
        returns = np.random.multivariate_normal([0.001, 0.0008], 
                                               [[0.0004, 0.0002], [0.0002, 0.0003]], 
                                               size=500)
        
        prices1 = 100 * np.cumprod(1 + returns[:, 0])
        prices2 = 80 * np.cumprod(1 + returns[:, 1])
        
        self.stock1_data = pd.DataFrame({
            'adjusted_close': prices1,
            'close': prices1,
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)
        
        self.stock2_data = pd.DataFrame({
            'adjusted_close': prices2,
            'close': prices2,
            'volume': np.random.randint(800000, 4000000, 500)
        }, index=dates)
    
    def test_multi_strategy_comparison(self):
        """Test comparison of multiple strategies on realistic data."""
        strategies = [
            DPTradingStrategy(max_transactions=1, transaction_cost=0.001),
            DPTradingStrategy(max_transactions=3, transaction_cost=0.001),
            DPTradingStrategy(max_transactions=5, transaction_cost=0.001),
            BuyAndHoldStrategy(transaction_cost=0.001)
        ]
        
        result = self.engine.strategy_comparison(strategies, self.stock1_data, "STOCK1")
        
        # All strategies should complete successfully
        assert len(result['individual_results']) == len(strategies)
        
        for strategy_name, strategy_result in result['individual_results'].items():
            assert 'error' not in strategy_result
            assert 'metrics' in strategy_result
            
            # Metrics should be reasonable
            metrics = strategy_result['metrics']
            assert isinstance(metrics.get('total_return'), (int, float))
            assert isinstance(metrics.get('sharpe_ratio'), (int, float))
            assert metrics.get('max_drawdown', 0) >= 0
    
    def test_performance_consistency(self):
        """Test that backtesting gives consistent results."""
        strategy = DPTradingStrategy(max_transactions=2, transaction_cost=0.001)
        
        # Run backtest multiple times
        results = []
        for _ in range(3):
            result = self.engine.simple_backtest(strategy, self.stock1_data, "STOCK1")
            results.append(result['metrics']['total_return'])
        
        # Results should be identical (deterministic algorithm)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-10
    
    def test_transaction_cost_impact(self):
        """Test impact of transaction costs on strategy performance."""
        strategies = [
            DPTradingStrategy(max_transactions=5, transaction_cost=0.0),
            DPTradingStrategy(max_transactions=5, transaction_cost=0.001),
            DPTradingStrategy(max_transactions=5, transaction_cost=0.01)
        ]
        
        results = []
        for strategy in strategies:
            result = self.engine.simple_backtest(strategy, self.stock1_data, "STOCK1")
            results.append(result['metrics']['total_return'])
        
        # Higher transaction costs should generally lead to lower returns
        # (unless the algorithm optimally reduces trading frequency)
        assert results[0] >= results[1]  # 0% cost >= 0.1% cost
        # Note: results[1] vs results[2] may not be monotonic due to optimization
    
    def test_data_quality_robustness(self):
        """Test robustness to data quality issues."""
        # Create data with some quality issues
        problematic_data = self.stock1_data.copy()
        
        # Add some NaN values
        problematic_data.loc[problematic_data.index[10:15], 'adjusted_close'] = np.nan
        
        # Add extreme outlier
        problematic_data.loc[problematic_data.index[50], 'adjusted_close'] = 1000000
        
        strategy = DPTradingStrategy(max_transactions=2)
        
        # Should handle problematic data gracefully
        try:
            result = self.engine.simple_backtest(strategy, problematic_data, "PROBLEMATIC")
            # If it completes, result should be structured correctly
            assert 'metrics' in result
        except Exception as e:
            # If it fails, should be a meaningful error
            assert isinstance(e, (ValueError, TypeError, IndexError))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup for edge case tests."""
        self.config = BacktestConfig()
        self.engine = BacktestEngine(self.config)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Very small dataset
        dates = pd.date_range('2023-01-01', periods=2, freq='D')
        small_data = pd.DataFrame({
            'adjusted_close': [100, 105],
            'close': [100, 105]
        }, index=dates)
        
        strategy = DPTradingStrategy(max_transactions=1)
        
        # Should handle gracefully
        result = self.engine.simple_backtest(strategy, small_data, "SMALL")
        assert 'metrics' in result
    
    def test_constant_prices(self):
        """Test with constant price data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        constant_data = pd.DataFrame({
            'adjusted_close': [100] * 100,
            'close': [100] * 100
        }, index=dates)
        
        strategy = DPTradingStrategy(max_transactions=5)
        result = self.engine.simple_backtest(strategy, constant_data, "CONSTANT")
        
        # Should result in no trades and zero profit
        assert result['metrics']['total_return'] == 0
        assert len(result['trades']) == 0
    
    def test_missing_columns(self):
        """Test behavior with missing required columns."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        incomplete_data = pd.DataFrame({
            'volume': [1000000] * 10  # Missing price columns
        }, index=dates)
        
        strategy = DPTradingStrategy(max_transactions=1)
        
        # Should raise appropriate error or handle gracefully
        with pytest.raises((KeyError, ValueError, AttributeError)):
            self.engine.simple_backtest(strategy, incomplete_data, "INCOMPLETE")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])