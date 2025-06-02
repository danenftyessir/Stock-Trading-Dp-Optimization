"""
Comprehensive Backtesting Framework for Trading Strategies.
Supports walk-forward analysis, out-of-sample testing, and performance evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models.dynamic_programing import DynamicProgrammingTrader
from ..models.baseline_strategies import BuyAndHoldStrategy, MovingAverageCrossoverStrategy
from ..analysis.performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    walk_forward: bool = True
    walk_forward_window: int = 252  # 1 year
    walk_forward_step: int = 63     # Quarter
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the strategy on training data."""
        pass
    
    @abstractmethod
    def predict_signals(self, data: pd.DataFrame) -> List[str]:
        """Generate trading signals for given data."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class DPTradingStrategy(TradingStrategy):
    """Dynamic Programming strategy wrapper for backtesting."""
    
    def __init__(self, max_transactions: int, transaction_cost: float = 0.001):
        self.max_transactions = max_transactions
        self.transaction_cost = transaction_cost
        self.trader = None
        
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit DP strategy (no fitting required, just initialize)."""
        self.trader = DynamicProgrammingTrader(
            max_transactions=self.max_transactions,
            transaction_cost=self.transaction_cost
        )
    
    def predict_signals(self, data: pd.DataFrame) -> List[str]:
        """Generate optimal trading signals using DP."""
        if 'adjusted_close' in data.columns:
            prices = data['adjusted_close'].tolist()
        else:
            prices = data['close'].tolist()
        
        max_profit, trades = self.trader.optimize_profit(prices)
        
        # Convert trades to signals
        signals = ['hold'] * len(prices)
        for trade in trades:
            if 'buy_day' in trade and trade['buy_day'] < len(signals):
                signals[trade['buy_day']] = 'buy'
            if 'sell_day' in trade and trade['sell_day'] < len(signals):
                signals[trade['sell_day']] = 'sell'
        
        return signals
    
    def get_name(self) -> str:
        return f"DP_K{self.max_transactions}"


class BacktestEngine:
    """
    Comprehensive backtesting engine with multiple evaluation modes.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config (BacktestConfig): Backtesting configuration
        """
        self.config = config
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=config.risk_free_rate
        )
        self.results_cache = {}
        
    def simple_backtest(self, strategy: TradingStrategy, 
                       data: pd.DataFrame,
                       symbol: str = "UNKNOWN") -> Dict:
        """
        Perform simple train/test backtest.
        
        Args:
            strategy (TradingStrategy): Trading strategy to test
            data (pd.DataFrame): Historical stock data
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Backtesting results
        """
        logger.info(f"Running simple backtest for {strategy.get_name()} on {symbol}")
        
        # Split data
        train_data, test_data = self._train_test_split(data)
        
        # Fit strategy on training data
        strategy.fit(train_data)
        
        # Generate signals for test data
        test_signals = strategy.predict_signals(test_data)
        
        # Execute trades based on signals
        portfolio_values, trades = self._execute_trades(
            test_data, test_signals, self.config.initial_capital
        )
        
        # Calculate performance metrics
        metrics = self.performance_analyzer.comprehensive_analysis(
            portfolio_values=portfolio_values,
            trades=trades,
            dates=test_data.index.strftime('%Y-%m-%d').tolist()
        )
        
        return {
            'strategy': strategy.get_name(),
            'symbol': symbol,
            'backtest_type': 'simple',
            'train_period': f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
            'test_period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
            'portfolio_values': portfolio_values,
            'trades': trades,
            'signals': test_signals,
            'metrics': metrics,
            'config': self.config.__dict__
        }
    
    def walk_forward_backtest(self, strategy: TradingStrategy,
                            data: pd.DataFrame,
                            symbol: str = "UNKNOWN") -> Dict:
        """
        Perform walk-forward backtesting.
        
        Args:
            strategy (TradingStrategy): Trading strategy to test
            data (pd.DataFrame): Historical stock data
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Walk-forward backtesting results
        """
        logger.info(f"Running walk-forward backtest for {strategy.get_name()} on {symbol}")
        
        window_size = self.config.walk_forward_window
        step_size = self.config.walk_forward_step
        
        if len(data) < window_size * 2:
            logger.warning(f"Insufficient data for walk-forward analysis")
            return self.simple_backtest(strategy, data, symbol)
        
        all_portfolio_values = []
        all_trades = []
        all_signals = []
        walk_forward_results = []
        
        # Start from the first full window
        start_idx = 0
        total_capital = self.config.initial_capital
        
        while start_idx + window_size + step_size <= len(data):
            # Define windows
            train_end = start_idx + window_size
            test_end = min(train_end + step_size, len(data))
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            logger.debug(f"Walk-forward period: {train_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Fit strategy on training data
            strategy.fit(train_data)
            
            # Generate signals for test period
            test_signals = strategy.predict_signals(test_data)
            
            # Execute trades for this period
            period_portfolio_values, period_trades = self._execute_trades(
                test_data, test_signals, total_capital
            )
            
            # Update capital for next period
            if period_portfolio_values:
                total_capital = period_portfolio_values[-1]
            
            # Store results
            all_portfolio_values.extend(period_portfolio_values)
            all_trades.extend(period_trades)
            all_signals.extend(test_signals)
            
            # Calculate period metrics
            period_metrics = self.performance_analyzer.comprehensive_analysis(
                portfolio_values=period_portfolio_values,
                trades=period_trades,
                dates=test_data.index.strftime('%Y-%m-%d').tolist()
            )
            
            walk_forward_results.append({
                'period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                'portfolio_values': period_portfolio_values,
                'trades': period_trades,
                'metrics': period_metrics
            })
            
            # Move to next period
            start_idx += step_size
        
        # Calculate overall metrics
        overall_metrics = self.performance_analyzer.comprehensive_analysis(
            portfolio_values=all_portfolio_values,
            trades=all_trades
        )
        
        return {
            'strategy': strategy.get_name(),
            'symbol': symbol,
            'backtest_type': 'walk_forward',
            'window_size': window_size,
            'step_size': step_size,
            'num_periods': len(walk_forward_results),
            'portfolio_values': all_portfolio_values,
            'trades': all_trades,
            'signals': all_signals,
            'metrics': overall_metrics,
            'period_results': walk_forward_results,
            'config': self.config.__dict__
        }
    
    def out_of_sample_test(self, strategy: TradingStrategy,
                          data: pd.DataFrame,
                          symbol: str = "UNKNOWN") -> Dict:
        """
        Perform rigorous out-of-sample testing with train/validation/test splits.
        
        Args:
            strategy (TradingStrategy): Trading strategy to test
            data (pd.DataFrame): Historical stock data
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Out-of-sample testing results
        """
        logger.info(f"Running out-of-sample test for {strategy.get_name()} on {symbol}")
        
        # Three-way split
        train_data, val_data, test_data = self._train_val_test_split(data)
        
        # Fit on training data
        strategy.fit(train_data)
        
        # Validate on validation set
        val_signals = strategy.predict_signals(val_data)
        val_portfolio_values, val_trades = self._execute_trades(
            val_data, val_signals, self.config.initial_capital
        )
        
        val_metrics = self.performance_analyzer.comprehensive_analysis(
            portfolio_values=val_portfolio_values,
            trades=val_trades,
            dates=val_data.index.strftime('%Y-%m-%d').tolist()
        )
        
        # Test on out-of-sample test set
        test_signals = strategy.predict_signals(test_data)
        test_portfolio_values, test_trades = self._execute_trades(
            test_data, test_signals, self.config.initial_capital
        )
        
        test_metrics = self.performance_analyzer.comprehensive_analysis(
            portfolio_values=test_portfolio_values,
            trades=test_trades,
            dates=test_data.index.strftime('%Y-%m-%d').tolist()
        )
        
        return {
            'strategy': strategy.get_name(),
            'symbol': symbol,
            'backtest_type': 'out_of_sample',
            'train_period': f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
            'validation_period': f"{val_data.index[0].date()} to {val_data.index[-1].date()}",
            'test_period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
            'validation_results': {
                'portfolio_values': val_portfolio_values,
                'trades': val_trades,
                'signals': val_signals,
                'metrics': val_metrics
            },
            'test_results': {
                'portfolio_values': test_portfolio_values,
                'trades': test_trades,
                'signals': test_signals,
                'metrics': test_metrics
            },
            'config': self.config.__dict__
        }
    
    def monte_carlo_backtest(self, strategy: TradingStrategy,
                           data: pd.DataFrame,
                           num_simulations: int = 1000,
                           symbol: str = "UNKNOWN") -> Dict:
        """
        Perform Monte Carlo backtesting with random data sampling.
        
        Args:
            strategy (TradingStrategy): Trading strategy to test
            data (pd.DataFrame): Historical stock data
            num_simulations (int): Number of Monte Carlo simulations
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Monte Carlo backtesting results
        """
        logger.info(f"Running Monte Carlo backtest for {strategy.get_name()} on {symbol}")
        
        simulation_results = []
        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for sim in range(num_simulations):
            if sim % 100 == 0:
                logger.debug(f"Monte Carlo simulation {sim}/{num_simulations}")
            
            # Create bootstrapped sample
            sample_data = self._bootstrap_sample(data)
            
            # Run simple backtest on sample
            try:
                result = self.simple_backtest(strategy, sample_data, f"{symbol}_sim{sim}")
                
                simulation_results.append(result)
                final_returns.append(result['metrics'].get('total_return', 0))
                max_drawdowns.append(result['metrics'].get('max_drawdown', 0))
                sharpe_ratios.append(result['metrics'].get('sharpe_ratio', 0))
                
            except Exception as e:
                logger.debug(f"Simulation {sim} failed: {e}")
                continue
        
        # Calculate Monte Carlo statistics
        mc_stats = {
            'num_simulations': len(simulation_results),
            'final_returns': {
                'mean': np.mean(final_returns),
                'std': np.std(final_returns),
                'min': np.min(final_returns),
                'max': np.max(final_returns),
                'percentiles': {
                    '5th': np.percentile(final_returns, 5),
                    '25th': np.percentile(final_returns, 25),
                    '50th': np.percentile(final_returns, 50),
                    '75th': np.percentile(final_returns, 75),
                    '95th': np.percentile(final_returns, 95)
                }
            },
            'max_drawdowns': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'max': np.max(max_drawdowns)
            },
            'sharpe_ratios': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'positive_ratio': np.sum(np.array(sharpe_ratios) > 0) / len(sharpe_ratios)
            }
        }
        
        return {
            'strategy': strategy.get_name(),
            'symbol': symbol,
            'backtest_type': 'monte_carlo',
            'monte_carlo_stats': mc_stats,
            'individual_simulations': simulation_results[:10],  # Store first 10 for analysis
            'config': self.config.__dict__
        }
    
    def strategy_comparison(self, strategies: List[TradingStrategy],
                          data: pd.DataFrame,
                          symbol: str = "UNKNOWN") -> Dict:
        """
        Compare multiple strategies on the same data.
        
        Args:
            strategies (List[TradingStrategy]): List of strategies to compare
            data (pd.DataFrame): Historical stock data
            symbol (str): Stock symbol for reporting
            
        Returns:
            Dict: Strategy comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies on {symbol}")
        
        comparison_results = {}
        
        for strategy in strategies:
            try:
                if self.config.walk_forward:
                    result = self.walk_forward_backtest(strategy, data, symbol)
                else:
                    result = self.simple_backtest(strategy, data, symbol)
                
                comparison_results[strategy.get_name()] = result
                
            except Exception as e:
                logger.error(f"Failed to backtest {strategy.get_name()}: {e}")
                comparison_results[strategy.get_name()] = {'error': str(e)}
        
        # Create comparison summary
        summary_metrics = []
        for strategy_name, result in comparison_results.items():
            if 'error' not in result:
                metrics = result.get('metrics', {})
                summary_metrics.append({
                    'strategy': strategy_name,
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'volatility': metrics.get('volatility', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'num_trades': metrics.get('number_of_trades', 0)
                })
        
        summary_df = pd.DataFrame(summary_metrics)
        
        return {
            'symbol': symbol,
            'comparison_type': 'strategy_comparison',
            'individual_results': comparison_results,
            'summary_table': summary_df.to_dict('records'),
            'best_strategy': {
                'by_return': summary_df.loc[summary_df['total_return'].idxmax()]['strategy'] if not summary_df.empty else None,
                'by_sharpe': summary_df.loc[summary_df['sharpe_ratio'].idxmax()]['strategy'] if not summary_df.empty else None,
                'by_drawdown': summary_df.loc[summary_df['max_drawdown'].idxmin()]['strategy'] if not summary_df.empty else None
            },
            'config': self.config.__dict__
        }
    
    def _train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_idx = int(len(data) * (self.config.train_ratio + self.config.validation_ratio))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        return train_data, test_data
    
    def _train_val_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        train_end = int(len(data) * self.config.train_ratio)
        val_end = int(len(data) * (self.config.train_ratio + self.config.validation_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def _bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create bootstrap sample of the data."""
        sample_indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[sample_indices].reset_index(drop=True)
    
    def _execute_trades(self, data: pd.DataFrame, signals: List[str], 
                       initial_capital: float) -> Tuple[List[float], List[Dict]]:
        """
        Execute trades based on signals and return portfolio values and trade details.
        
        Args:
            data (pd.DataFrame): Price data
            signals (List[str]): Trading signals ('buy', 'sell', 'hold')
            initial_capital (float): Starting capital
            
        Returns:
            Tuple[List[float], List[Dict]]: Portfolio values and trade records
        """
        if 'adjusted_close' in data.columns:
            prices = data['adjusted_close'].tolist()
        else:
            prices = data['close'].tolist()
        
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 'out'  # 'in' or 'out' of market
        
        for day, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 'buy' and position == 'out' and cash > 0:
                # Buy shares
                shares_to_buy = cash / (price * (1 + self.config.transaction_cost))
                cost = shares_to_buy * price * (1 + self.config.transaction_cost)
                
                if cost <= cash:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'in'
                    
                    trades.append({
                        'action': 'buy',
                        'day': day,
                        'date': data.index[day] if day < len(data) else None,
                        'price': price,
                        'shares': shares_to_buy,
                        'cost': cost,
                        'remaining_cash': cash
                    })
                    
            elif signal == 'sell' and position == 'in' and shares > 0:
                # Sell shares
                proceeds = shares * price * (1 - self.config.transaction_cost)
                cash += proceeds
                
                # Calculate profit for this trade
                last_buy = None
                for trade in reversed(trades):
                    if trade['action'] == 'buy':
                        last_buy = trade
                        break
                
                profit = proceeds - last_buy['cost'] if last_buy else 0
                
                trades.append({
                    'action': 'sell',
                    'day': day,
                    'date': data.index[day] if day < len(data) else None,
                    'price': price,
                    'shares': shares,
                    'proceeds': proceeds,
                    'profit': profit,
                    'remaining_cash': cash
                })
                
                shares = 0
                position = 'out'
            
            # Calculate current portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        # Process trades into buy-sell pairs
        processed_trades = self._process_trades_for_analysis(trades)
        
        return portfolio_values, processed_trades
    
    def _process_trades_for_analysis(self, raw_trades: List[Dict]) -> List[Dict]:
        """Process raw trades into buy-sell pairs for analysis."""
        processed = []
        buy_trade = None
        
        for trade in raw_trades:
            if trade['action'] == 'buy':
                buy_trade = trade
            elif trade['action'] == 'sell' and buy_trade is not None:
                processed.append({
                    'buy_day': buy_trade['day'],
                    'buy_date': buy_trade.get('date'),
                    'buy_price': buy_trade['price'],
                    'sell_day': trade['day'],
                    'sell_date': trade.get('date'),
                    'sell_price': trade['price'],
                    'shares': trade['shares'],
                    'profit': trade['profit'],
                    'duration': trade['day'] - buy_trade['day'],
                    'return_pct': (trade['price'] - buy_trade['price']) / buy_trade['price']
                })
                buy_trade = None
                
        return processed


class BacktestRunner:
    """
    High-level interface for running comprehensive backtests.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.engine = BacktestEngine(config)
        
    def run_dp_parameter_sweep(self, data: pd.DataFrame,
                              k_values: List[int],
                              symbol: str = "UNKNOWN") -> Dict:
        """
        Run parameter sweep for DP strategy with different k values.
        
        Args:
            data (pd.DataFrame): Stock data
            k_values (List[int]): List of k values to test
            symbol (str): Stock symbol
            
        Returns:
            Dict: Parameter sweep results
        """
        logger.info(f"Running DP parameter sweep for {symbol} with k values: {k_values}")
        
        strategies = [DPTradingStrategy(k, self.config.transaction_cost) for k in k_values]
        
        results = self.engine.strategy_comparison(strategies, data, symbol)
        
        # Add parameter-specific analysis
        k_analysis = {}
        for k in k_values:
            strategy_name = f"DP_K{k}"
            if strategy_name in results['individual_results']:
                result = results['individual_results'][strategy_name]
                if 'error' not in result:
                    k_analysis[k] = {
                        'total_return': result['metrics'].get('total_return', 0),
                        'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0),
                        'max_drawdown': result['metrics'].get('max_drawdown', 0),
                        'num_trades': result['metrics'].get('number_of_trades', 0)
                    }
        
        results['k_analysis'] = k_analysis
        results['optimal_k'] = {
            'by_return': max(k_analysis.keys(), key=lambda k: k_analysis[k]['total_return']) if k_analysis else None,
            'by_sharpe': max(k_analysis.keys(), key=lambda k: k_analysis[k]['sharpe_ratio']) if k_analysis else None
        }
        
        return results
    
    def run_comprehensive_analysis(self, data: pd.DataFrame,
                                 symbol: str = "UNKNOWN") -> Dict:
        """
        Run comprehensive backtesting analysis including multiple strategies and tests.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            
        Returns:
            Dict: Comprehensive analysis results
        """
        logger.info(f"Running comprehensive backtesting analysis for {symbol}")
        
        # Define strategies to test
        strategies = [
            DPTradingStrategy(2, self.config.transaction_cost),
            DPTradingStrategy(5, self.config.transaction_cost),
            DPTradingStrategy(10, self.config.transaction_cost),
            BuyAndHoldStrategy(self.config.transaction_cost),
            MovingAverageCrossoverStrategy(20, 50, self.config.transaction_cost)
        ]
        
        results = {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_period': f"{data.index[0].date()} to {data.index[-1].date()}",
            'total_days': len(data)
        }
        
        # Strategy comparison
        comparison_result = self.engine.strategy_comparison(strategies, data, symbol)
        results['strategy_comparison'] = comparison_result
        
        # Out-of-sample testing for best DP strategy
        best_dp_strategy = DPTradingStrategy(5, self.config.transaction_cost)  # Use k=5 as example
        oos_result = self.engine.out_of_sample_test(best_dp_strategy, data, symbol)
        results['out_of_sample_test'] = oos_result
        
        # Walk-forward analysis
        if self.config.walk_forward and len(data) >= self.config.walk_forward_window * 2:
            wf_result = self.engine.walk_forward_backtest(best_dp_strategy, data, symbol)
            results['walk_forward_analysis'] = wf_result
        
        return results