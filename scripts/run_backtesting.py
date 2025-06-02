#!/usr/bin/env python3
"""
Backtesting Script for Stock Trading Optimization.
Runs comprehensive backtesting analysis on trading strategies.

Usage:
    python scripts/run_backtesting.py --strategy dp --k-values 2 5 10
    python scripts/run_backtesting.py --strategy all --symbols AAPL GOOGL
    python scripts/run_backtesting.py --optimization --k-range 1 20
"""

import sys
import os
import argparse
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.backtesting import BacktestEngine, BacktestConfig, DPTradingStrategy
from optimization.parameter_tuning import DPParameterOptimizer, OptimizationConfig
from models.baseline_strategies import BuyAndHoldStrategy, MovingAverageCrossoverStrategy, MomentumStrategy
from analysis.performance_metrics import PerformanceAnalyzer
from utils.logger import setup_global_logging
from utils.helpers import load_config, save_json, ensure_directory


def setup_logging():
    """Setup logging for the script."""
    log_config = {
        'log_level': 'INFO',  # Fixed: Changed from 'level' to 'log_level'
        'enable_console': True,
        'enable_file': True,
        'log_dir': 'logs',
        'log_file': f'data_collection_{datetime.now().strftime("%Y%m%d")}.log'
    }
    return setup_global_logging(log_config)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtesting analysis for trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --strategy dp --k-values 2 5 10     # Test DP with specific k values
  %(prog)s --strategy all --symbols AAPL       # Test all strategies on AAPL
  %(prog)s --optimization --k-range 1 20       # Optimize k parameter
  %(prog)s --walk-forward --window 252         # Walk-forward analysis
  %(prog)s --monte-carlo --simulations 1000    # Monte Carlo backtesting
        """
    )
    
    # Strategy selection
    parser.add_argument(
        '--strategy', 
        choices=['dp', 'baseline', 'all'],
        default='all',
        help='Strategy type to test (default: all)'
    )
    parser.add_argument(
        '--k-values', 
        nargs='+', 
        type=int,
        default=[2, 5, 10],
        help='K values for DP strategy (default: 2 5 10)'
    )
    
    # Symbol selection
    parser.add_argument(
        '--symbols', 
        nargs='+',
        help='Stock symbols to analyze (default: from config)'
    )
    parser.add_argument(
        '--exclude', 
        nargs='+',
        help='Symbols to exclude from analysis'
    )
    
    # Backtesting modes
    parser.add_argument(
        '--walk-forward', 
        action='store_true',
        help='Enable walk-forward analysis'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=252,
        help='Walk-forward window size in days (default: 252)'
    )
    parser.add_argument(
        '--step', 
        type=int, 
        default=63,
        help='Walk-forward step size in days (default: 63)'
    )
    
    # Optimization
    parser.add_argument(
        '--optimization', 
        action='store_true',
        help='Run parameter optimization'
    )
    parser.add_argument(
        '--k-range', 
        nargs=2, 
        type=int, 
        default=[1, 20],
        help='K parameter range for optimization (default: 1 20)'
    )
    parser.add_argument(
        '--objective', 
        choices=['total_return', 'sharpe_ratio', 'calmar_ratio'],
        default='sharpe_ratio',
        help='Optimization objective metric (default: sharpe_ratio)'
    )
    
    # Monte Carlo
    parser.add_argument(
        '--monte-carlo', 
        action='store_true',
        help='Run Monte Carlo backtesting'
    )
    parser.add_argument(
        '--simulations', 
        type=int, 
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    parser.add_argument(
        '--initial-capital', 
        type=float, 
        default=100000,
        help='Initial capital for backtesting (default: 100000)'
    )
    parser.add_argument(
        '--transaction-cost', 
        type=float, 
        default=0.001,
        help='Transaction cost as decimal (default: 0.001 = 0.1%%)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', 
        default='results/backtesting',
        help='Output directory for results (default: results/backtesting)'
    )
    parser.add_argument(
        '--save-details', 
        action='store_true',
        help='Save detailed trade logs and portfolio values'
    )
    parser.add_argument(
        '--generate-plots', 
        action='store_true',
        help='Generate performance plots (requires matplotlib)'
    )
    
    # Control options
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Validate setup without running backtests'
    )
    
    return parser.parse_args()


def load_stock_data(symbols: list, data_dir: str = "data/processed/model_inputs") -> dict:
    """Load processed stock data for specified symbols."""
    stock_data = {}
    data_path = Path(data_dir)
    
    for symbol in symbols:
        file_path = data_path / f"{symbol}_daily.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Validate required columns
                required_cols = ['adjusted_close']
                if not all(col in df.columns for col in required_cols):
                    logging.warning(f"Missing required columns in {symbol} data")
                    continue
                
                # Filter out any invalid data
                df = df.dropna(subset=['adjusted_close'])
                
                if len(df) < 100:  # Minimum data requirement
                    logging.warning(f"Insufficient data for {symbol}: {len(df)} days")
                    continue
                
                stock_data[symbol] = df
                logging.info(f"Loaded {symbol}: {len(df)} days of data")
                
            except Exception as e:
                logging.error(f"Failed to load data for {symbol}: {e}")
        else:
            logging.warning(f"Data file not found for {symbol}: {file_path}")
    
    return stock_data


def create_strategies(k_values: list, transaction_cost: float) -> dict:
    """Create strategy instances for backtesting."""
    strategies = {}
    
    # Dynamic Programming strategies
    for k in k_values:
        strategies[f"DP_K{k}"] = DPTradingStrategy(k, transaction_cost)
    
    # Baseline strategies
    strategies["BuyHold"] = BuyAndHoldStrategy(transaction_cost)
    strategies["MA_Crossover"] = MovingAverageCrossoverStrategy(20, 50, transaction_cost)
    strategies["Momentum"] = MomentumStrategy(10, 0.02, transaction_cost)
    
    return strategies


def run_single_backtest(engine: BacktestEngine, strategy, data: pd.DataFrame, 
                       symbol: str, backtest_type: str = "simple") -> dict:
    """Run a single backtest and return results."""
    try:
        if backtest_type == "walk_forward":
            result = engine.walk_forward_backtest(strategy, data, symbol)
        elif backtest_type == "out_of_sample":
            result = engine.out_of_sample_test(strategy, data, symbol)
        else:
            result = engine.simple_backtest(strategy, data, symbol)
        
        return result
        
    except Exception as e:
        logging.error(f"Backtest failed for {strategy.get_name()} on {symbol}: {e}")
        return {'error': str(e), 'strategy': strategy.get_name(), 'symbol': symbol}


def run_optimization(optimizer: DPParameterOptimizer, data: pd.DataFrame, 
                    symbol: str, k_range: tuple) -> dict:
    """Run parameter optimization for a single symbol."""
    try:
        logging.info(f"Optimizing parameters for {symbol} (k range: {k_range})")
        
        result = optimizer.optimize_k_parameter(
            data=data,
            k_range=k_range,
            symbol=symbol
        )
        
        if result.get('best_parameters'):
            best_k = result['best_parameters']['k']
            best_score = result['best_score']
            logging.info(f"✅ {symbol}: Optimal k={best_k}, Score={best_score:.4f}")
        else:
            logging.warning(f"⚠️ {symbol}: Optimization failed")
        
        return result
        
    except Exception as e:
        logging.error(f"Optimization failed for {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol}


def generate_summary_report(results: dict, output_dir: str):
    """Generate summary report from backtesting results."""
    summary_data = []
    
    for symbol, symbol_results in results.items():
        if isinstance(symbol_results, dict) and 'strategies' in symbol_results:
            for strategy_name, strategy_result in symbol_results['strategies'].items():
                if 'error' not in strategy_result:
                    metrics = strategy_result.get('metrics', {})
                    
                    summary_data.append({
                        'Symbol': symbol,
                        'Strategy': strategy_name,
                        'Total_Return': metrics.get('total_return', 0),
                        'Annualized_Return': metrics.get('annualized_return', 0),
                        'Volatility': metrics.get('volatility', 0),
                        'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                        'Max_Drawdown': metrics.get('max_drawdown', 0),
                        'Number_of_Trades': metrics.get('number_of_trades', 0),
                        'Win_Rate': metrics.get('win_rate', 0)
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        summary_file = Path(output_dir) / "backtesting_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Generate analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_backtests': len(summary_data),
            'symbols_analyzed': summary_df['Symbol'].nunique(),
            'strategies_tested': summary_df['Strategy'].nunique(),
            'best_performers': {
                'by_return': summary_df.loc[summary_df['Total_Return'].idxmax()].to_dict(),
                'by_sharpe': summary_df.loc[summary_df['Sharpe_Ratio'].idxmax()].to_dict(),
                'by_drawdown': summary_df.loc[summary_df['Max_Drawdown'].idxmin()].to_dict()
            },
            'strategy_averages': summary_df.groupby('Strategy')[
                ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown']
            ].mean().to_dict()
        }
        
        analysis_file = Path(output_dir) / "backtesting_analysis.json"
        save_json(analysis, analysis_file)
        
        logging.info(f"Summary report saved to {summary_file}")
        logging.info(f"Analysis report saved to {analysis_file}")
        
        return summary_df, analysis
    
    return None, None


def print_results_summary(results: dict):
    """Print formatted results summary to console."""
    print("\n" + "="*80)
    print("BACKTESTING RESULTS SUMMARY")
    print("="*80)
    
    total_tests = 0
    successful_tests = 0
    
    for symbol, symbol_results in results.items():
        if isinstance(symbol_results, dict):
            if 'strategies' in symbol_results:
                # Strategy comparison results
                strategies = symbol_results['strategies']
                symbol_total = len(strategies)
                symbol_success = len([s for s in strategies.values() if 'error' not in s])
                
                print(f"\n📊 {symbol}:")
                print(f"   Tests run: {symbol_total}, Successful: {symbol_success}")
                
                # Show best performers
                valid_strategies = {k: v for k, v in strategies.items() if 'error' not in v}
                if valid_strategies:
                    best_return = max(valid_strategies.items(), 
                                    key=lambda x: x[1].get('metrics', {}).get('total_return', -float('inf')))
                    best_sharpe = max(valid_strategies.items(), 
                                    key=lambda x: x[1].get('metrics', {}).get('sharpe_ratio', -float('inf')))
                    
                    print(f"   Best Return: {best_return[0]} ({best_return[1]['metrics']['total_return']:.2%})")
                    print(f"   Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.3f})")
                
                total_tests += symbol_total
                successful_tests += symbol_success
                
            elif 'optimization' in symbol_results:
                # Optimization results
                opt_result = symbol_results['optimization']
                if 'best_parameters' in opt_result:
                    best_k = opt_result['best_parameters']['k']
                    best_score = opt_result['best_score']
                    print(f"\n🎯 {symbol} Optimization:")
                    print(f"   Optimal k: {best_k}, Score: {best_score:.4f}")
                else:
                    print(f"\n❌ {symbol} Optimization: Failed")
    
    print(f"\n" + "="*80)
    print(f"OVERALL SUMMARY")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("BACKTESTING SCRIPT STARTED")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Determine symbols to analyze
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            symbols = config['data']['tickers']
        
        # Apply exclusions
        if args.exclude:
            exclude_symbols = [s.upper() for s in args.exclude]
            symbols = [s for s in symbols if s not in exclude_symbols]
            logger.info(f"Excluded symbols: {exclude_symbols}")
        
        logger.info(f"Analyzing symbols: {symbols}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run mode - validating setup only")
            stock_data = load_stock_data(symbols)
            logger.info(f"✅ Would analyze {len(stock_data)} symbols with valid data")
            logger.info(f"✅ Setup validation complete")
            return 0
        
        # Load stock data
        stock_data = load_stock_data(symbols)
        if not stock_data:
            logger.error("No valid stock data found")
            return 1
        
        logger.info(f"Loaded data for {len(stock_data)} symbols")
        
        # Setup output directory
        output_dir = ensure_directory(args.output_dir)
        session_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_directory(session_dir)
        
        # Configure backtesting
        backtest_config = BacktestConfig(
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            walk_forward=args.walk_forward,
            walk_forward_window=args.window,
            walk_forward_step=args.step
        )
        
        engine = BacktestEngine(backtest_config)
        results = {}
        
        # Run optimization if requested
        if args.optimization:
            logger.info("Running parameter optimization")
            
            opt_config = OptimizationConfig(
                objective_metric=args.objective,
                optimization_method='grid_search'
            )
            
            optimizer = DPParameterOptimizer(opt_config, backtest_config)
            
            for symbol, data in stock_data.items():
                opt_result = run_optimization(optimizer, data, symbol, tuple(args.k_range))
                results[symbol] = {'optimization': opt_result}
        
        # Run backtesting
        else:
            logger.info("Running backtesting analysis")
            
            # Create strategies
            strategies = {}
            if args.strategy in ['dp', 'all']:
                for k in args.k_values:
                    strategies[f"DP_K{k}"] = DPTradingStrategy(k, args.transaction_cost)
            
            if args.strategy in ['baseline', 'all']:
                strategies.update({
                    "BuyHold": BuyAndHoldStrategy(args.transaction_cost),
                    "MA_Crossover": MovingAverageCrossoverStrategy(20, 50, args.transaction_cost),
                    "Momentum": MomentumStrategy(10, 0.02, args.transaction_cost)
                })
            
            logger.info(f"Testing {len(strategies)} strategies")
            
            # Run backtests for each symbol
            for symbol, data in stock_data.items():
                logger.info(f"Backtesting {symbol}...")
                
                symbol_results = {'strategies': {}}
                
                for strategy_name, strategy in strategies.items():
                    logger.info(f"  Testing {strategy_name}")
                    
                    # Determine backtest type
                    if args.monte_carlo:
                        result = engine.monte_carlo_backtest(
                            strategy, data, args.simulations, symbol
                        )
                    elif args.walk_forward:
                        result = engine.walk_forward_backtest(strategy, data, symbol)
                    else:
                        result = engine.simple_backtest(strategy, data, symbol)
                    
                    symbol_results['strategies'][strategy_name] = result
                    
                    # Log basic results
                    if 'error' not in result:
                        metrics = result.get('metrics', {})
                        total_return = metrics.get('total_return', 0)
                        sharpe_ratio = metrics.get('sharpe_ratio', 0)
                        logger.info(f"    Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.3f}")
                
                results[symbol] = symbol_results
        
        # Save detailed results
        results_file = session_dir / "detailed_results.json"
        save_json(results, results_file)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Generate summary report
        if not args.optimization:
            summary_df, analysis = generate_summary_report(results, session_dir)
            
            if summary_df is not None:
                logger.info("Summary report generated successfully")
            else:
                logger.warning("No valid results for summary report")
        
        # Print results summary
        print_results_summary(results)
        
        # Generate plots if requested
        if args.generate_plots:
            try:
                from visualization.performance_plots import PerformancePlotter
                
                plotter = PerformancePlotter()
                plots_dir = session_dir / "plots"
                ensure_directory(plots_dir)
                
                logger.info(f"Generating performance plots in: {plots_dir}")
                # Implementation would generate various plots here
                
            except ImportError:
                logger.warning("Plotting libraries not available, skipping plot generation")
        
        logger.info("Backtesting script completed successfully")
        logger.info(f"Results saved in: {session_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())