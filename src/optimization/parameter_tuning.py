"""
Parameter Optimization Module for Trading Strategies.
FIXED VERSION - Added proper sensitivity analysis output generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from datetime import datetime
import logging
from itertools import product
from dataclasses import dataclass
import json
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid

try:
    from optimization.backtesting import BacktestEngine
    from analysis.performance_metrics import PerformanceAnalyzer
except ImportError:
    # Alternative import for script execution
    import sys
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent if current_dir.name == 'optimization' else current_dir.parent / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from optimization.backtesting import BacktestEngine
    from analysis.performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    objective_metric: str = 'sharpe_ratio'  # 'total_return', 'sharpe_ratio', 'calmar_ratio'
    optimization_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    n_trials: int = 100  # For random search and Bayesian optimization
    cv_folds: int = 3  # Cross-validation folds
    test_ratio: float = 0.2  # Hold-out test set ratio
    min_improvement: float = 0.001  # Minimum improvement threshold
    max_time_minutes: int = 60  # Maximum optimization time
    parallel: bool = False  # Parallel execution (future enhancement)


class ParameterOptimizer:
    """
    Generic parameter optimizer for trading strategies.
    """
    
    def __init__(self, config: OptimizationConfig, backtest_config):
        """
        Initialize parameter optimizer.
        
        Args:
            config (OptimizationConfig): Optimization configuration
            backtest_config: Backtesting configuration
        """
        self.config = config
        self.backtest_config = backtest_config
        
        # Import here to avoid circular imports
        try:
            from optimization.backtesting import BacktestEngine
            from analysis.performance_metrics import PerformanceAnalyzer
        except ImportError:
            # Alternative import for script execution
            import sys
            from pathlib import Path
            
            current_dir = Path(__file__).parent
            src_dir = current_dir.parent if current_dir.name == 'optimization' else current_dir.parent / 'src'
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from optimization.backtesting import BacktestEngine
            from analysis.performance_metrics import PerformanceAnalyzer
        
        self.backtest_engine = BacktestEngine(backtest_config)
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
    def objective_function(self, params: Dict, data: pd.DataFrame, 
                          symbol: str = "UNKNOWN") -> float:
        """
        Objective function to optimize.
        
        Args:
            params (Dict): Parameter dictionary
            data (pd.DataFrame): Training data
            symbol (str): Stock symbol
            
        Returns:
            float: Objective score
        """
        try:
            # Create strategy with given parameters
            strategy = self._create_strategy(params)
            
            # Run backtest
            result = self.backtest_engine.simple_backtest(strategy, data, symbol)
            
            # Extract objective metric
            metrics = result.get('metrics', {})
            score = metrics.get(self.config.objective_metric, 0.0)
            
            # Handle invalid results
            if np.isnan(score) or np.isinf(score):
                score = float('-inf')
            
            # FIXED: Cap score to reasonable range based on metric type
            if self.config.objective_metric == 'sharpe_ratio':
                score = max(-5.0, min(5.0, score))
            elif self.config.objective_metric == 'total_return':
                score = max(-0.99, min(10.0, score))  # Cap between -99% and 1000%
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"New best score: {score:.4f} with params: {params}")
            
            return score
            
        except Exception as e:
            logger.warning(f"Objective function failed with params {params}: {e}")
            return float('-inf')
    
    def _create_strategy(self, params: Dict):
        """Create strategy instance with given parameters."""
        raise NotImplementedError("Subclasses must implement _create_strategy")
    
    def grid_search(self, param_grid: Dict, data: pd.DataFrame, 
                   symbol: str = "UNKNOWN") -> Dict:
        """
        Perform grid search optimization.
        
        Args:
            param_grid (Dict): Parameter grid specification
            data (pd.DataFrame): Training data
            symbol (str): Stock symbol
            
        Returns:
            Dict: Optimization results
        """
        logger.info(f"Starting grid search optimization for {symbol}")
        logger.info(f"Parameter grid: {param_grid}")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        logger.info(f"Total combinations to evaluate: {total_combinations}")
        
        start_time = datetime.now()
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            # Check time limit
            if (datetime.now() - start_time).total_seconds() > self.config.max_time_minutes * 60:
                logger.warning(f"Time limit reached. Evaluated {i}/{total_combinations} combinations.")
                break
            
            self.objective_function(params, data, symbol)
        
        return self._compile_results('grid_search', symbol)
    
    def random_search(self, param_distributions: Dict, data: pd.DataFrame,
                     symbol: str = "UNKNOWN") -> Dict:
        """
        Perform random search optimization.
        
        Args:
            param_distributions (Dict): Parameter distribution specifications
            data (pd.DataFrame): Training data
            symbol (str): Stock symbol
            
        Returns:
            Dict: Optimization results
        """
        logger.info(f"Starting random search optimization for {symbol}")
        logger.info(f"Parameter distributions: {param_distributions}")
        
        start_time = datetime.now()
        
        for trial in range(self.config.n_trials):
            if trial % 20 == 0:
                logger.info(f"Trial {trial}/{self.config.n_trials}")
            
            # Check time limit
            if (datetime.now() - start_time).total_seconds() > self.config.max_time_minutes * 60:
                logger.warning(f"Time limit reached. Completed {trial}/{self.config.n_trials} trials.")
                break
            
            # Sample random parameters
            params = self._sample_random_params(param_distributions)
            
            self.objective_function(params, data, symbol)
        
        return self._compile_results('random_search', symbol)
    
    def _sample_random_params(self, param_distributions: Dict) -> Dict:
        """Sample random parameters from distributions."""
        params = {}
        
        for param_name, distribution in param_distributions.items():
            if isinstance(distribution, tuple) and len(distribution) == 2:
                # Uniform distribution (min, max)
                low, high = distribution
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = np.random.randint(low, high + 1)
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            elif isinstance(distribution, list):
                # Discrete choice
                params[param_name] = np.random.choice(distribution)
            
            else:
                raise ValueError(f"Unsupported distribution type for parameter {param_name}")
        
        return params
    
    def _compile_results(self, method: str, symbol: str) -> Dict:
        """Compile optimization results."""
        if not self.optimization_history:
            return {'error': 'No successful evaluations'}
        
        # Sort history by score
        sorted_history = sorted(self.optimization_history, 
                              key=lambda x: x['score'], reverse=True)
        
        # Calculate statistics
        scores = [entry['score'] for entry in self.optimization_history if entry['score'] != float('-inf')]
        
        results = {
            'symbol': symbol,
            'method': method,
            'objective_metric': self.config.objective_metric,
            'optimization_timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.optimization_history),
            'successful_evaluations': len(scores),
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'top_10_results': sorted_history[:10],
            'score_statistics': {
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'min': np.min(scores) if scores else 0,
                'max': np.max(scores) if scores else 0,
                'median': np.median(scores) if scores else 0
            },
            'config': {
                'optimization': self.config.__dict__,
                'backtest': self.backtest_config.__dict__
            }
        }
        
        return results


class DPParameterOptimizer(ParameterOptimizer):
    """
    Specialized optimizer for Dynamic Programming strategy parameters.
    FIXED: Added proper sensitivity analysis implementation.
    """
    
    def _create_strategy(self, params: Dict):
        """Create DP strategy with given parameters."""
        try:
            from optimization.backtesting import DPTradingStrategy
        except ImportError:
            # Alternative import for script execution
            import sys
            from pathlib import Path
            
            current_dir = Path(__file__).parent
            src_dir = current_dir.parent if current_dir.name == 'optimization' else current_dir.parent / 'src'
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from optimization.backtesting import DPTradingStrategy
        
        return DPTradingStrategy(
            max_transactions=params.get('k', 2),
            transaction_cost=params.get('transaction_cost', self.backtest_config.transaction_cost)
        )
    
    def optimize_k_parameter(self, data: pd.DataFrame, 
                           k_range: Tuple[int, int] = (1, 20),
                           symbol: str = "UNKNOWN") -> Dict:
        """
        Optimize the k parameter for DP strategy.
        
        Args:
            data (pd.DataFrame): Stock data
            k_range (Tuple[int, int]): Range of k values to test
            symbol (str): Stock symbol
            
        Returns:
            Dict: Optimization results
        """
        param_grid = {
            'k': list(range(k_range[0], k_range[1] + 1))
        }
        
        return self.grid_search(param_grid, data, symbol)
    
    def sensitivity_analysis(self, data: pd.DataFrame,
                           base_params: Dict,
                           param_ranges: Dict,
                           symbol: str = "UNKNOWN") -> Dict:
        """
        Perform sensitivity analysis around base parameters.
        FIXED: Enhanced sensitivity analysis with proper output generation.
        
        Args:
            data (pd.DataFrame): Stock data
            base_params (Dict): Base parameter values
            param_ranges (Dict): Ranges for sensitivity analysis
            symbol (str): Stock symbol
            
        Returns:
            Dict: Sensitivity analysis results
        """
        logger.info(f"Starting sensitivity analysis for {symbol}")
        
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            if param_name not in base_params:
                continue
            
            param_scores = []
            param_values = []
            param_details = []
            
            # Test different values for this parameter
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous parameter
                test_values = np.linspace(param_range[0], param_range[1], 11)
            else:
                # Discrete parameter
                test_values = param_range
            
            for value in test_values:
                test_params = base_params.copy()
                test_params[param_name] = value
                
                score = self.objective_function(test_params, data, symbol)
                
                param_values.append(value)
                param_scores.append(score)
                
                # Store detailed results
                if self.optimization_history:
                    param_details.append(self.optimization_history[-1])
            
            sensitivity_results[param_name] = {
                'values': param_values,
                'scores': param_scores,
                'details': param_details,
                'best_value': param_values[np.argmax(param_scores)] if param_scores else base_params[param_name],
                'best_score': max(param_scores) if param_scores else 0,
                'sensitivity': np.std(param_scores) if param_scores else 0,  # Higher std = more sensitive
                'range_impact': (max(param_scores) - min(param_scores)) if param_scores else 0
            }
        
        # FIXED: Save sensitivity analysis results to proper directory
        self._save_sensitivity_results(sensitivity_results, symbol)
        
        return {
            'symbol': symbol,
            'base_parameters': base_params,
            'sensitivity_results': sensitivity_results,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': self._generate_sensitivity_summary(sensitivity_results)
        }
    
    def _save_sensitivity_results(self, sensitivity_results: Dict, symbol: str):
        """
        Save sensitivity analysis results to the proper directory.
        FIXED: Ensure output directory exists and save results.
        """
        try:
            # Create output directory
            output_dir = Path("results/sensitivity_analysis/k_optimization")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_file = output_dir / f"{symbol}_k_sensitivity.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for param_name, param_data in sensitivity_results.items():
                serializable_results[param_name] = {
                    'values': [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for v in param_data['values']],
                    'scores': [float(s) if isinstance(s, (np.integer, np.floating)) else s 
                              for s in param_data['scores']],
                    'best_value': float(param_data['best_value']) if isinstance(param_data['best_value'], (np.integer, np.floating)) else param_data['best_value'],
                    'best_score': float(param_data['best_score']),
                    'sensitivity': float(param_data['sensitivity']),
                    'range_impact': float(param_data['range_impact'])
                }
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Sensitivity analysis results saved to {results_file}")
            
            # Save summary CSV
            summary_file = output_dir / f"{symbol}_k_optimization_summary.csv"
            self._save_sensitivity_summary_csv(sensitivity_results, summary_file, symbol)
            
        except Exception as e:
            logger.error(f"Failed to save sensitivity results for {symbol}: {e}")
    
    def _save_sensitivity_summary_csv(self, sensitivity_results: Dict, file_path: Path, symbol: str):
        """Save sensitivity analysis summary as CSV."""
        try:
            summary_data = []
            
            for param_name, param_data in sensitivity_results.items():
                for i, (value, score) in enumerate(zip(param_data['values'], param_data['scores'])):
                    summary_data.append({
                        'Symbol': symbol,
                        'Parameter': param_name,
                        'Value': value,
                        'Score': score,
                        'Is_Best': value == param_data['best_value']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(file_path, index=False)
                logger.info(f"Sensitivity summary CSV saved to {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to save sensitivity summary CSV: {e}")
    
    def _generate_sensitivity_summary(self, sensitivity_results: Dict) -> Dict:
        """Generate summary statistics for sensitivity analysis."""
        summary = {
            'parameters_analyzed': list(sensitivity_results.keys()),
            'most_sensitive_parameter': None,
            'least_sensitive_parameter': None,
            'optimal_parameters': {}
        }
        
        if sensitivity_results:
            # Find most and least sensitive parameters
            sensitivities = {param: data['sensitivity'] for param, data in sensitivity_results.items()}
            
            if sensitivities:
                summary['most_sensitive_parameter'] = max(sensitivities.keys(), key=lambda k: sensitivities[k])
                summary['least_sensitive_parameter'] = min(sensitivities.keys(), key=lambda k: sensitivities[k])
            
            # Extract optimal parameters
            for param, data in sensitivity_results.items():
                summary['optimal_parameters'][param] = data['best_value']
        
        return summary
    
    def comprehensive_k_analysis(self, data: pd.DataFrame, 
                                symbol: str = "UNKNOWN",
                                k_range: Tuple[int, int] = (1, 20)) -> Dict:
        """
        Perform comprehensive analysis of k parameter impact.
        FIXED: Complete analysis with sensitivity and visualization data.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            k_range (Tuple[int, int]): Range of k values to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive k analysis for {symbol}")
        
        # Step 1: Basic optimization
        optimization_result = self.optimize_k_parameter(data, k_range, symbol)
        
        # Step 2: Sensitivity analysis around optimal k
        if optimization_result.get('best_parameters'):
            base_k = optimization_result['best_parameters']['k']
            
            # Create sensitivity range around optimal k
            sensitivity_range = list(range(max(1, base_k - 5), min(k_range[1] + 1, base_k + 6)))
            
            sensitivity_result = self.sensitivity_analysis(
                data=data,
                base_params={'k': base_k},
                param_ranges={'k': sensitivity_range},
                symbol=symbol
            )
        else:
            sensitivity_result = {}
        
        # Step 3: Detailed analysis of each k value
        k_analysis_details = {}
        for k in range(k_range[0], k_range[1] + 1):
            try:
                strategy = self._create_strategy({'k': k})
                result = self.backtest_engine.simple_backtest(strategy, data, symbol)
                
                k_analysis_details[k] = {
                    'metrics': result.get('metrics', {}),
                    'trades': result.get('trades', []),
                    'portfolio_values': result.get('portfolio_values', [])
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze k={k} for {symbol}: {e}")
                k_analysis_details[k] = {'error': str(e)}
        
        # Compile comprehensive results
        comprehensive_results = {
            'symbol': symbol,
            'k_range': k_range,
            'optimization_result': optimization_result,
            'sensitivity_analysis': sensitivity_result,
            'detailed_k_analysis': k_analysis_details,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': self._generate_comprehensive_summary(k_analysis_details, optimization_result)
        }
        
        # Save comprehensive results
        self._save_comprehensive_results(comprehensive_results, symbol)
        
        return comprehensive_results
    
    def _generate_comprehensive_summary(self, k_details: Dict, optimization_result: Dict) -> Dict:
        """Generate comprehensive summary of k analysis."""
        summary = {
            'optimal_k': optimization_result.get('best_parameters', {}).get('k', 'unknown'),
            'optimal_score': optimization_result.get('best_score', 0),
            'k_performance_trend': {},
            'trading_frequency_analysis': {}
        }
        
        # Analyze performance trend across k values
        k_scores = {}
        k_trades = {}
        
        for k, details in k_details.items():
            if 'error' not in details:
                metrics = details.get('metrics', {})
                trades = details.get('trades', [])
                
                score = metrics.get('total_return', 0)
                k_scores[k] = score
                k_trades[k] = len(trades)
        
        if k_scores:
            summary['k_performance_trend'] = {
                'best_k_by_return': max(k_scores.keys(), key=lambda k: k_scores[k]),
                'worst_k_by_return': min(k_scores.keys(), key=lambda k: k_scores[k]),
                'performance_range': max(k_scores.values()) - min(k_scores.values()),
                'average_performance': np.mean(list(k_scores.values()))
            }
        
        if k_trades:
            summary['trading_frequency_analysis'] = {
                'avg_trades_per_k': np.mean(list(k_trades.values())),
                'max_trades': max(k_trades.values()),
                'min_trades': min(k_trades.values()),
                'k_with_most_trades': max(k_trades.keys(), key=lambda k: k_trades[k])
            }
        
        return summary
    
    def _save_comprehensive_results(self, results: Dict, symbol: str):
        """Save comprehensive analysis results."""
        try:
            output_dir = Path("results/sensitivity_analysis/k_optimization")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_dir / f"{symbol}_comprehensive_k_analysis.json"
            
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Comprehensive k analysis saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save comprehensive results for {symbol}: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


def run_k_sensitivity_study(stock_data: Dict[str, pd.DataFrame],
                           k_range: Tuple[int, int] = (1, 20)) -> Dict:
    """
    Run comprehensive k parameter sensitivity study across multiple stocks.
    FIXED: Generate proper output in sensitivity analysis directory.
    
    Args:
        stock_data (Dict[str, pd.DataFrame]): Stock data dictionary
        k_range (Tuple[int, int]): Range of k values to study
        
    Returns:
        Dict: Study results
    """
    logger.info(f"Starting k sensitivity study for {len(stock_data)} stocks")
    
    # Create configurations
    opt_config = OptimizationConfig(objective_metric='sharpe_ratio')
    
    # Import here to avoid circular imports
    from ..optimization.backtesting import BacktestConfig
    backtest_config = BacktestConfig()
    
    study_results = {
        'study_timestamp': datetime.now().isoformat(),
        'stocks_analyzed': list(stock_data.keys()),
        'k_range': k_range,
        'individual_analyses': {},
        'cross_stock_summary': {}
    }
    
    # Run analysis for each stock
    for symbol, data in stock_data.items():
        logger.info(f"Analyzing k sensitivity for {symbol}")
        
        try:
            optimizer = DPParameterOptimizer(opt_config, backtest_config)
            
            # Run comprehensive k analysis
            analysis_result = optimizer.comprehensive_k_analysis(
                data=data,
                symbol=symbol,
                k_range=k_range
            )
            
            study_results['individual_analyses'][symbol] = analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
            study_results['individual_analyses'][symbol] = {'error': str(e)}
    
    # Generate cross-stock summary
    study_results['cross_stock_summary'] = _generate_cross_stock_summary(
        study_results['individual_analyses']
    )
    
    # Save study results
    _save_study_results(study_results)
    
    logger.info("K sensitivity study completed")
    return study_results


def _generate_cross_stock_summary(individual_analyses: Dict) -> Dict:
    """Generate summary across all stocks."""
    summary = {
        'optimal_k_distribution': {},
        'average_performance_by_k': {},
        'most_consistent_k': None
    }
    
    optimal_ks = []
    k_performances = {}
    
    for symbol, analysis in individual_analyses.items():
        if 'error' not in analysis:
            optimal_k = analysis.get('summary', {}).get('optimal_k')
            if optimal_k and optimal_k != 'unknown':
                optimal_ks.append(optimal_k)
            
            # Collect performance by k
            k_details = analysis.get('detailed_k_analysis', {})
            for k, details in k_details.items():
                if 'error' not in details:
                    metrics = details.get('metrics', {})
                    performance = metrics.get('total_return', 0)
                    
                    if k not in k_performances:
                        k_performances[k] = []
                    k_performances[k].append(performance)
    
    # Analyze optimal k distribution
    if optimal_ks:
        from collections import Counter
        k_counts = Counter(optimal_ks)
        summary['optimal_k_distribution'] = dict(k_counts)
        summary['most_common_optimal_k'] = k_counts.most_common(1)[0][0]
    
    # Average performance by k
    for k, performances in k_performances.items():
        summary['average_performance_by_k'][k] = {
            'mean': np.mean(performances),
            'std': np.std(performances),
            'count': len(performances)
        }
    
    return summary


def _save_study_results(study_results: Dict):
    """Save study results to file."""
    try:
        output_dir = Path("results/sensitivity_analysis/k_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        study_file = output_dir / f"k_sensitivity_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Make JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        serializable_results = make_serializable(study_results)
        
        with open(study_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Study results saved to {study_file}")
        
    except Exception as e:
        logger.error(f"Failed to save study results: {e}")