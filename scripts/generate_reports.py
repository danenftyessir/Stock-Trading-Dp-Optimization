#!/usr/bin/env python3
"""
Report Generation Script for Stock Trading Optimization.
Generates comprehensive analysis reports from backtesting results.

Usage:
    python scripts/generate_reports.py --input results/backtesting/20241201_143022
    python scripts/generate_reports.py --latest --format html pdf
    python scripts/generate_reports.py --comparison --symbols AAPL GOOGL
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logger import setup_global_logging
from utils.helpers import load_json, save_json, ensure_directory
from analysis.performance_metrics import PerformanceAnalyzer
from visualization.performance_plots import PerformancePlotter, InteractivePlotter


def setup_logging():
    """Setup logging for the script."""
    log_config = {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_file': True,
        'log_dir': 'logs',
        'log_file': f'report_generation_{datetime.now().strftime("%Y%m%d")}.log'
    }
    return setup_global_logging(log_config)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive analysis reports from backtesting results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input results/backtesting/20241201_143022  # Specific results
  %(prog)s --latest                                     # Latest results
  %(prog)s --comparison --symbols AAPL GOOGL           # Compare symbols
  %(prog)s --format html pdf --template executive       # Multiple formats
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', 
        help='Path to backtesting results directory'
    )
    input_group.add_argument(
        '--latest', 
        action='store_true',
        help='Use latest backtesting results'
    )
    
    # Report types
    parser.add_argument(
        '--type', 
        choices=['performance', 'comparison', 'optimization', 'risk', 'all'],
        default='all',
        help='Type of report to generate (default: all)'
    )
    
    # Output formats
    parser.add_argument(
        '--format', 
        nargs='+',
        choices=['html', 'pdf', 'markdown', 'excel'],
        default=['html', 'markdown'],
        help='Output formats (default: html markdown)'
    )
    
    # Report templates
    parser.add_argument(
        '--template', 
        choices=['executive', 'technical', 'research'],
        default='technical',
        help='Report template (default: technical)'
    )
    
    # Analysis options
    parser.add_argument(
        '--symbols', 
        nargs='+',
        help='Specific symbols to include in report'
    )
    parser.add_argument(
        '--strategies', 
        nargs='+',
        help='Specific strategies to include in report'
    )
    parser.add_argument(
        '--comparison', 
        action='store_true',
        help='Generate strategy comparison analysis'
    )
    
    # Content options
    parser.add_argument(
        '--include-plots', 
        action='store_true',
        default=True,
        help='Include performance plots (default: True)'
    )
    parser.add_argument(
        '--include-tables', 
        action='store_true',
        default=True,
        help='Include summary tables (default: True)'
    )
    parser.add_argument(
        '--include-raw-data', 
        action='store_true',
        help='Include raw backtesting data'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', 
        default='reports',
        help='Output directory for reports (default: reports)'
    )
    parser.add_argument(
        '--title', 
        help='Custom report title'
    )
    parser.add_argument(
        '--author', 
        default='Stock Trading Optimization System',
        help='Report author'
    )
    
    # Control options
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--open-browser', 
        action='store_true',
        help='Open generated HTML report in browser'
    )
    
    return parser.parse_args()


def find_latest_results(results_base_dir: str = "results/backtesting") -> Optional[Path]:
    """Find the most recent backtesting results directory."""
    results_path = Path(results_base_dir)
    
    if not results_path.exists():
        return None
    
    # Look for timestamped directories
    timestamped_dirs = [
        d for d in results_path.iterdir() 
        if d.is_dir() and len(d.name) == 15 and d.name.replace('_', '').isdigit()
    ]
    
    if not timestamped_dirs:
        return None
    
    # Return the most recent
    return max(timestamped_dirs, key=lambda x: x.name)


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load backtesting results from directory."""
    results_data = {}
    
    # Load main results file
    results_file = results_dir / "detailed_results.json"
    if results_file.exists():
        results_data['detailed_results'] = load_json(results_file)
    
    # Load summary if available
    summary_file = results_dir / "backtesting_summary.csv"
    if summary_file.exists():
        results_data['summary_table'] = pd.read_csv(summary_file)
    
    # Load analysis if available
    analysis_file = results_dir / "backtesting_analysis.json"
    if analysis_file.exists():
        results_data['analysis'] = load_json(analysis_file)
    
    return results_data


class ReportGenerator:
    """Generate comprehensive analysis reports from backtesting results."""
    
    def __init__(self, results_data: Dict[str, Any], output_dir: str):
        self.results_data = results_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.plotter = PerformancePlotter()
        
        # Extract data for easier access
        self.detailed_results = results_data.get('detailed_results', {})
        self.summary_table = results_data.get('summary_table')
        self.analysis = results_data.get('analysis', {})
    
    def generate_performance_report(self, symbols: Optional[List[str]] = None) -> Dict:
        """Generate performance analysis report."""
        logging.info("Generating performance report")
        
        report_data = {
            'title': 'Performance Analysis Report',
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': [],
            'strategies_tested': [],
            'performance_summary': {},
            'detailed_metrics': {}
        }
        
        # Process results for each symbol
        for symbol, symbol_data in self.detailed_results.items():
            if symbols and symbol not in symbols:
                continue
            
            report_data['symbols_analyzed'].append(symbol)
            
            if 'strategies' in symbol_data:
                symbol_performance = {}
                
                for strategy_name, strategy_result in symbol_data['strategies'].items():
                    if 'error' not in strategy_result:
                        metrics = strategy_result.get('metrics', {})
                        
                        # Extract key performance metrics
                        performance_metrics = {
                            'total_return': metrics.get('total_return', 0),
                            'annualized_return': metrics.get('annualized_return', 0),
                            'volatility': metrics.get('volatility', 0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                            'max_drawdown': metrics.get('max_drawdown', 0),
                            'number_of_trades': metrics.get('number_of_trades', 0),
                            'win_rate': metrics.get('win_rate', 0)
                        }
                        
                        symbol_performance[strategy_name] = performance_metrics
                        
                        if strategy_name not in report_data['strategies_tested']:
                            report_data['strategies_tested'].append(strategy_name)
                
                report_data['detailed_metrics'][symbol] = symbol_performance
        
        # Calculate aggregated statistics
        if self.summary_table is not None:
            agg_stats = {}
            
            for strategy in report_data['strategies_tested']:
                strategy_data = self.summary_table[
                    self.summary_table['Strategy'] == strategy
                ]
                
                if not strategy_data.empty:
                    agg_stats[strategy] = {
                        'mean_return': strategy_data['Total_Return'].mean(),
                        'mean_sharpe': strategy_data['Sharpe_Ratio'].mean(),
                        'mean_drawdown': strategy_data['Max_Drawdown'].mean(),
                        'std_return': strategy_data['Total_Return'].std(),
                        'best_symbol': strategy_data.loc[
                            strategy_data['Total_Return'].idxmax(), 'Symbol'
                        ],
                        'worst_symbol': strategy_data.loc[
                            strategy_data['Total_Return'].idxmin(), 'Symbol'
                        ]
                    }
            
            report_data['performance_summary'] = agg_stats
        
        return report_data
    
    def generate_comparison_report(self) -> Dict:
        """Generate strategy comparison report."""
        logging.info("Generating strategy comparison report")
        
        if self.summary_table is None:
            return {'error': 'Summary table not available for comparison'}
        
        comparison_data = {
            'title': 'Strategy Comparison Report',
            'timestamp': datetime.now().isoformat(),
            'strategy_rankings': {},
            'pairwise_comparisons': {},
            'statistical_analysis': {}
        }
        
        # Strategy rankings by different metrics
        metrics = ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown']
        
        for metric in metrics:
            if metric in self.summary_table.columns:
                if metric == 'Max_Drawdown':
                    # Lower is better for drawdown
                    rankings = self.summary_table.groupby('Strategy')[metric].mean().sort_values()
                else:
                    # Higher is better for returns and Sharpe
                    rankings = self.summary_table.groupby('Strategy')[metric].mean().sort_values(ascending=False)
                
                comparison_data['strategy_rankings'][metric] = rankings.to_dict()
        
        # Pairwise strategy comparisons
        strategies = self.summary_table['Strategy'].unique()
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies[i+1:], i+1):
                
                strategy1_data = self.summary_table[self.summary_table['Strategy'] == strategy1]
                strategy2_data = self.summary_table[self.summary_table['Strategy'] == strategy2]
                
                # Find common symbols
                common_symbols = set(strategy1_data['Symbol']) & set(strategy2_data['Symbol'])
                
                if common_symbols:
                    comparison_key = f"{strategy1}_vs_{strategy2}"
                    
                    # Compare on common symbols
                    strategy1_returns = []
                    strategy2_returns = []
                    
                    for symbol in common_symbols:
                        ret1 = strategy1_data[strategy1_data['Symbol'] == symbol]['Total_Return'].iloc[0]
                        ret2 = strategy2_data[strategy2_data['Symbol'] == symbol]['Total_Return'].iloc[0]
                        
                        strategy1_returns.append(ret1)
                        strategy2_returns.append(ret2)
                    
                    # Calculate comparison statistics
                    win_rate = np.mean(np.array(strategy1_returns) > np.array(strategy2_returns))
                    avg_difference = np.mean(np.array(strategy1_returns) - np.array(strategy2_returns))
                    
                    comparison_data['pairwise_comparisons'][comparison_key] = {
                        'win_rate': win_rate,
                        'average_difference': avg_difference,
                        'common_symbols': len(common_symbols)
                    }
        
        return comparison_data
    
    def generate_risk_report(self) -> Dict:
        """Generate risk analysis report."""
        logging.info("Generating risk analysis report")
        
        risk_data = {
            'title': 'Risk Analysis Report',
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {},
            'risk_rankings': {},
            'correlation_analysis': {}
        }
        
        # Extract risk metrics for each strategy/symbol combination
        all_returns = {}
        
        for symbol, symbol_data in self.detailed_results.items():
            if 'strategies' in symbol_data:
                for strategy_name, strategy_result in symbol_data['strategies'].items():
                    if 'error' not in strategy_result:
                        portfolio_values = strategy_result.get('portfolio_values', [])
                        
                        if portfolio_values:
                            returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
                            
                            key = f"{strategy_name}_{symbol}"
                            all_returns[key] = returns
                            
                            # Calculate additional risk metrics
                            risk_metrics = {
                                'volatility': np.std(returns) * np.sqrt(252),
                                'skewness': pd.Series(returns).skew(),
                                'kurtosis': pd.Series(returns).kurtosis(),
                                'var_95': np.percentile(returns, 5),
                                'var_99': np.percentile(returns, 1),
                                'downside_deviation': np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
                            }
                            
                            risk_data['risk_metrics'][key] = risk_metrics
        
        # Risk rankings
        if risk_data['risk_metrics']:
            volatilities = {k: v['volatility'] for k, v in risk_data['risk_metrics'].items()}
            risk_data['risk_rankings']['volatility'] = dict(
                sorted(volatilities.items(), key=lambda x: x[1])
            )
        
        return risk_data
    
    def create_visualizations(self, report_data: Dict) -> Dict[str, str]:
        """Create visualization plots for the report."""
        logging.info("Creating visualizations")
        
        plots_dir = self.output_dir / "figures"
        plots_dir.mkdir(exist_ok=True)
        
        plot_files = {}
        
        try:
            # Performance comparison plot
            if self.summary_table is not None:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Returns by strategy
                strategy_returns = self.summary_table.groupby('Strategy')['Total_Return'].mean()
                axes[0, 0].bar(strategy_returns.index, strategy_returns.values)
                axes[0, 0].set_title('Average Total Return by Strategy')
                axes[0, 0].set_ylabel('Total Return')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Sharpe ratio by strategy
                strategy_sharpe = self.summary_table.groupby('Strategy')['Sharpe_Ratio'].mean()
                axes[0, 1].bar(strategy_sharpe.index, strategy_sharpe.values)
                axes[0, 1].set_title('Average Sharpe Ratio by Strategy')
                axes[0, 1].set_ylabel('Sharpe Ratio')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Risk-return scatter
                axes[1, 0].scatter(self.summary_table['Volatility'], 
                                 self.summary_table['Total_Return'],
                                 alpha=0.6)
                axes[1, 0].set_xlabel('Volatility')
                axes[1, 0].set_ylabel('Total Return')
                axes[1, 0].set_title('Risk-Return Profile')
                
                # Drawdown distribution
                axes[1, 1].hist(self.summary_table['Max_Drawdown'], bins=20, alpha=0.7)
                axes[1, 1].set_xlabel('Maximum Drawdown')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Drawdown Distribution')
                
                plt.tight_layout()
                
                plot_file = plots_dir / "performance_overview.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files['performance_overview'] = str(plot_file)
            
            # Strategy comparison heatmap
            if self.summary_table is not None and len(self.summary_table['Strategy'].unique()) > 1:
                pivot_table = self.summary_table.pivot_table(
                    index='Symbol', 
                    columns='Strategy', 
                    values='Total_Return'
                )
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
                plt.title('Total Returns Heatmap by Strategy and Symbol')
                plt.tight_layout()
                
                plot_file = plots_dir / "strategy_heatmap.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files['strategy_heatmap'] = str(plot_file)
            
            logging.info(f"Created {len(plot_files)} visualization plots")
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {e}")
        
        return plot_files
    
    def generate_html_report(self, report_data: Dict, plot_files: Dict[str, str]) -> str:
        """Generate HTML report."""
        logging.info("Generating HTML report")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data.get('title', 'Trading Analysis Report')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e9e9e9; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data.get('title', 'Trading Analysis Report')}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Period:</strong> {report_data.get('analysis_period', 'See detailed results')}</p>
    </div>
"""
        
        # Executive summary
        if 'performance_summary' in report_data:
            html_content += """
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
"""
            
            summary = report_data['performance_summary']
            if summary:
                best_strategy = max(summary.keys(), 
                                  key=lambda k: summary[k].get('mean_return', 0))
                best_return = summary[best_strategy]['mean_return']
                
                html_content += f"""
            <div class="metric">
                <div class="metric-label">Best Performing Strategy</div>
                <div class="metric-value">{best_strategy}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Average Return</div>
                <div class="metric-value">{best_return:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Strategies Tested</div>
                <div class="metric-value">{len(summary)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Symbols Analyzed</div>
                <div class="metric-value">{len(report_data.get('symbols_analyzed', []))}</div>
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        # Add plots if available
        for plot_name, plot_file in plot_files.items():
            html_content += f"""
    <div class="section">
        <h3>{plot_name.replace('_', ' ').title()}</h3>
        <div class="plot">
            <img src="figures/{Path(plot_file).name}" alt="{plot_name}">
        </div>
    </div>
"""
        
        # Summary table
        if self.summary_table is not None:
            html_content += """
    <div class="section">
        <h2>Detailed Results Summary</h2>
        <table>
            <thead>
                <tr>
"""
            
            for col in self.summary_table.columns:
                html_content += f"                    <th>{col}</th>\n"
            
            html_content += """
                </tr>
            </thead>
            <tbody>
"""
            
            for _, row in self.summary_table.iterrows():
                html_content += "                <tr>\n"
                for col in self.summary_table.columns:
                    value = row[col]
                    if isinstance(value, float):
                        if 'Return' in col or 'Ratio' in col:
                            formatted_value = f"{value:.3f}"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"                    <td>{formatted_value}</td>\n"
                html_content += "                </tr>\n"
            
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        html_content += """
    <div class="footer">
        <p><em>This report was generated by the Stock Trading Optimization System.</em></p>
        <p><em>Disclaimer: Past performance does not guarantee future results. This analysis is for educational purposes only.</em></p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_file = self.output_dir / "trading_analysis_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML report saved to: {html_file}")
        return str(html_file)
    
    def generate_markdown_report(self, report_data: Dict) -> str:
        """Generate Markdown report."""
        logging.info("Generating Markdown report")
        
        md_content = f"""# {report_data.get('title', 'Trading Analysis Report')}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Period:** {report_data.get('analysis_period', 'See detailed results')}

## Executive Summary

"""
        
        if 'performance_summary' in report_data:
            summary = report_data['performance_summary']
            if summary:
                best_strategy = max(summary.keys(), 
                                  key=lambda k: summary[k].get('mean_return', 0))
                best_return = summary[best_strategy]['mean_return']
                
                md_content += f"""
- **Best Performing Strategy:** {best_strategy}
- **Best Average Return:** {best_return:.2%}
- **Strategies Tested:** {len(summary)}
- **Symbols Analyzed:** {len(report_data.get('symbols_analyzed', []))}

### Strategy Performance Summary

| Strategy | Avg Return | Avg Sharpe | Avg Drawdown |
|----------|------------|------------|--------------|
"""
                
                for strategy, metrics in summary.items():
                    md_content += f"| {strategy} | {metrics['mean_return']:.2%} | {metrics['mean_sharpe']:.3f} | {metrics['mean_drawdown']:.2%} |\n"
        
        # Detailed results table
        if self.summary_table is not None:
            md_content += "\n## Detailed Results\n\n"
            md_content += self.summary_table.to_markdown(index=False, floatfmt=".3f")
        
        md_content += """

## Methodology

This analysis was conducted using:

- **Dynamic Programming Algorithm:** Optimal k-transaction profit maximization
- **Backtesting Framework:** Walk-forward analysis with out-of-sample testing
- **Risk Metrics:** Sharpe ratio, maximum drawdown, volatility analysis
- **Data Source:** Alpha Vantage API historical OHLCV data

## Disclaimer

*This analysis is for educational and research purposes only. Past performance does not guarantee future results. Trading involves risk of financial loss. Always conduct your own research and consider consulting with financial professionals before making investment decisions.*

---

*Report generated by Stock Trading Optimization System*
"""
        
        # Save Markdown report
        md_file = self.output_dir / "trading_analysis_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logging.info(f"Markdown report saved to: {md_file}")
        return str(md_file)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("REPORT GENERATION SCRIPT STARTED")
    logger.info("=" * 60)
    
    try:
        # Determine input directory
        if args.latest:
            input_dir = find_latest_results()
            if input_dir is None:
                logger.error("No backtesting results found")
                return 1
            logger.info(f"Using latest results: {input_dir}")
        else:
            input_dir = Path(args.input)
            if not input_dir.exists():
                logger.error(f"Input directory not found: {input_dir}")
                return 1
        
        # Load results data
        logger.info("Loading backtesting results")
        results_data = load_results(input_dir)
        
        if not results_data:
            logger.error("No valid results data found")
            return 1
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_directory(args.output_dir) / f"report_{timestamp}"
        ensure_directory(output_dir)
        
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize report generator
        generator = ReportGenerator(results_data, output_dir)
        
        # Generate different types of reports
        all_report_data = {}
        
        if args.type in ['performance', 'all']:
            performance_data = generator.generate_performance_report(args.symbols)
            all_report_data['performance'] = performance_data
        
        if args.type in ['comparison', 'all'] or args.comparison:
            comparison_data = generator.generate_comparison_report()
            all_report_data['comparison'] = comparison_data
        
        if args.type in ['risk', 'all']:
            risk_data = generator.generate_risk_report()
            all_report_data['risk'] = risk_data
        
        # Create visualizations
        plot_files = {}
        if args.include_plots:
            plot_files = generator.create_visualizations(all_report_data)
        
        # Combine all report data
        combined_report_data = {
            'title': args.title or 'Stock Trading Optimization Analysis Report',
            'author': args.author,
            'timestamp': datetime.now().isoformat(),
            'input_source': str(input_dir),
            **all_report_data
        }
        
        # Generate reports in requested formats
        generated_files = []
        
        if 'html' in args.format:
            html_file = generator.generate_html_report(combined_report_data, plot_files)
            generated_files.append(html_file)
        
        if 'markdown' in args.format:
            md_file = generator.generate_markdown_report(combined_report_data)
            generated_files.append(md_file)
        
        if 'excel' in args.format and generator.summary_table is not None:
            excel_file = output_dir / "trading_analysis_data.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                generator.summary_table.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add other data sheets if available
                if 'performance' in all_report_data:
                    perf_data = all_report_data['performance']
                    if 'performance_summary' in perf_data:
                        perf_df = pd.DataFrame(perf_data['performance_summary']).T
                        perf_df.to_excel(writer, sheet_name='Performance_Summary')
            
            generated_files.append(str(excel_file))
            logger.info(f"Excel report saved to: {excel_file}")
        
        # Save combined report data
        save_json(combined_report_data, output_dir / "report_data.json")
        
        # Print summary
        print(f"\n📊 Report Generation Complete")
        print("=" * 50)
        print(f"Output directory: {output_dir}")
        print(f"Generated files:")
        for file_path in generated_files:
            print(f"  • {Path(file_path).name}")
        
        if plot_files:
            print(f"Visualizations:")
            for plot_name in plot_files.keys():
                print(f"  • {plot_name}.png")
        
        # Open browser if requested
        if args.open_browser and 'html' in args.format:
            try:
                import webbrowser
                html_file = next((f for f in generated_files if f.endswith('.html')), None)
                if html_file:
                    webbrowser.open(f"file://{Path(html_file).absolute()}")
                    logger.info("Opened HTML report in browser")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
        
        logger.info("Report generation completed successfully")
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