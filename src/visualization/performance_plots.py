"""
Performance Visualization Module.
Creates comprehensive charts and plots for trading strategy analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class PerformancePlotter:
    """
    Comprehensive plotting tools for trading strategy performance analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'default'):
        """
        Initialize performance plotter.
        
        Args:
            figsize (Tuple[int, int]): Default figure size
            style (str): Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_portfolio_performance(self, portfolio_values: List[float],
                                 benchmark_values: Optional[List[float]] = None,
                                 dates: Optional[List[str]] = None,
                                 title: str = "Portfolio Performance",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio performance over time with optional benchmark comparison.
        
        Args:
            portfolio_values (List[float]): Portfolio values over time
            benchmark_values (Optional[List[float]]): Benchmark values for comparison
            dates (Optional[List[str]]): Date labels
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_axis = dates if dates else range(len(portfolio_values))
        if dates:
            x_axis = pd.to_datetime(dates)
        
        # Plot portfolio performance
        ax.plot(x_axis, portfolio_values, label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            ax.plot(x_axis, benchmark_values, label='Benchmark', linewidth=2, 
                   color='red', alpha=0.7, linestyle='--')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date' if dates else 'Time Period', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        return fig
    
    def plot_returns_distribution(self, portfolio_values: List[float],
                                title: str = "Returns Distribution",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of returns with statistics.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        # Calculate returns
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Returns Histogram')
        axes[0, 0].set_xlabel('Daily Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        stats_text = f'Mean: {mean_return:.4f}\nStd: {std_return:.4f}\nSkew: {skewness:.2f}\nKurt: {kurtosis:.2f}'
        axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Q-Q plot
        from scipy import stats as scipy_stats
        scipy_stats.probplot(returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(returns, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 0].set_title('Returns Box Plot')
        axes[1, 0].set_ylabel('Daily Returns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
        axes[1, 1].plot(cumulative_returns, color='green', linewidth=2)
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_xlabel('Time Period')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution plot saved to {save_path}")
        
        return fig
    
    def plot_drawdown_analysis(self, portfolio_values: List[float],
                             dates: Optional[List[str]] = None,
                             title: str = "Drawdown Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown analysis including underwater curve.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            dates (Optional[List[str]]): Date labels
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        # Calculate drawdowns
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        x_axis = dates if dates else range(len(portfolio_values))
        if dates:
            x_axis = pd.to_datetime(dates)
        
        # Portfolio value and peaks
        ax1.plot(x_axis, portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
        ax1.plot(x_axis, peak, label='Peak Value', color='red', linestyle='--', alpha=0.7)
        ax1.fill_between(x_axis, portfolio_values, peak, alpha=0.3, color='red', label='Drawdown')
        
        ax1.set_title('Portfolio Value and Drawdown Periods')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Underwater curve (drawdown percentage)
        ax2.fill_between(x_axis, 0, -drawdown * 100, alpha=0.7, color='red')
        ax2.plot(x_axis, -drawdown * 100, color='darkred', linewidth=1)
        
        ax2.set_title('Underwater Curve (Drawdown %)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date' if dates else 'Time Period')
        ax2.grid(True, alpha=0.3)
        
        # Add max drawdown annotation
        max_dd_idx = np.argmax(drawdown)
        max_dd_value = drawdown[max_dd_idx] * 100
        
        if dates:
            max_dd_date = pd.to_datetime(dates[max_dd_idx])
            ax2.annotate(f'Max DD: {max_dd_value:.2f}%', 
                        xy=(max_dd_date, -max_dd_value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Format x-axis for dates
        if dates:
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown analysis plot saved to {save_path}")
        
        return fig
    
    def plot_rolling_metrics(self, portfolio_values: List[float],
                           rolling_window: int = 252,
                           dates: Optional[List[str]] = None,
                           title: str = "Rolling Performance Metrics",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            rolling_window (int): Rolling window size
            dates (Optional[List[str]]): Date labels
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if len(portfolio_values) < rolling_window + 1:
            logger.warning(f"Insufficient data for rolling analysis (need >{rolling_window} points)")
            return plt.figure()
        
        # Calculate rolling metrics
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        rolling_return = []
        rolling_volatility = []
        rolling_sharpe = []
        
        for i in range(rolling_window, len(portfolio_values)):
            window_values = portfolio_values[i-rolling_window:i+1]
            window_returns = returns[i-rolling_window:i]
            
            # Annualized return
            annual_return = ((window_values[-1] / window_values[0]) ** (252 / rolling_window)) - 1
            rolling_return.append(annual_return)
            
            # Annualized volatility
            volatility = np.std(window_returns) * np.sqrt(252)
            rolling_volatility.append(volatility)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
            rolling_sharpe.append(sharpe)
        
        # Prepare x-axis
        if dates:
            rolling_dates = pd.to_datetime(dates[rolling_window:])
            x_axis = rolling_dates
        else:
            x_axis = range(rolling_window, len(portfolio_values))
        
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Rolling returns
        axes[0].plot(x_axis, np.array(rolling_return) * 100, color='blue', linewidth=2)
        axes[0].set_title(f'Rolling {rolling_window}-Day Annualized Return')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Rolling volatility
        axes[1].plot(x_axis, np.array(rolling_volatility) * 100, color='orange', linewidth=2)
        axes[1].set_title(f'Rolling {rolling_window}-Day Annualized Volatility')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        axes[2].plot(x_axis, rolling_sharpe, color='green', linewidth=2)
        axes[2].set_title(f'Rolling {rolling_window}-Day Sharpe Ratio')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].set_xlabel('Date' if dates else 'Time Period')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Format x-axis for dates
        if dates:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.xticks(rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling metrics plot saved to {save_path}")
        
        return fig
    
    def plot_strategy_comparison(self, strategies_data: Dict[str, Dict],
                               title: str = "Strategy Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple trading strategies.
        
        Args:
            strategies_data (Dict[str, Dict]): Dictionary with strategy names and their data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        strategy_names = list(strategies_data.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategy_names)))
        
        # Portfolio values comparison
        for i, (strategy_name, data) in enumerate(strategies_data.items()):
            portfolio_values = data.get('portfolio_values', [])
            if portfolio_values:
                # Normalize to start at 100
                normalized_values = np.array(portfolio_values) / portfolio_values[0] * 100
                axes[0, 0].plot(normalized_values, label=strategy_name, 
                              color=colors[i], linewidth=2)
        
        axes[0, 0].set_title('Normalized Portfolio Performance')
        axes[0, 0].set_ylabel('Portfolio Value (Base=100)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns comparison
        returns_data = []
        for strategy_name, data in strategies_data.items():
            portfolio_values = data.get('portfolio_values', [])
            if len(portfolio_values) > 1:
                total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
                returns_data.append(total_return)
            else:
                returns_data.append(0)
        
        bars = axes[0, 1].bar(strategy_names, returns_data, color=colors, alpha=0.7)
        axes[0, 1].set_title('Total Returns Comparison')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns_data):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Risk-Return scatter
        sharpe_ratios = []
        volatilities = []
        
        for strategy_name, data in strategies_data.items():
            portfolio_values = data.get('portfolio_values', [])
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
                vol = np.std(returns) * np.sqrt(252) * 100
                mean_return = np.mean(returns) * 252 * 100
                sharpe = (mean_return - 2) / vol if vol > 0 else 0  # Assuming 2% risk-free rate
                
                volatilities.append(vol)
                sharpe_ratios.append(sharpe)
            else:
                volatilities.append(0)
                sharpe_ratios.append(0)
        
        scatter = axes[1, 0].scatter(volatilities, [r for r in returns_data], 
                                   c=colors, s=100, alpha=0.7)
        
        for i, strategy_name in enumerate(strategy_names):
            axes[1, 0].annotate(strategy_name, 
                              (volatilities[i], returns_data[i]),
                              xytext=(5, 5), textcoords='offset points')
        
        axes[1, 0].set_xlabel('Volatility (%)')
        axes[1, 0].set_ylabel('Total Return (%)')
        axes[1, 0].set_title('Risk-Return Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sharpe ratio comparison
        bars = axes[1, 1].bar(strategy_names, sharpe_ratios, color=colors, alpha=0.7)
        axes[1, 1].set_title('Sharpe Ratio Comparison')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (0.01 if value >= 0 else -0.03),
                           f'{value:.2f}', ha='center', 
                           va='bottom' if value >= 0 else 'top')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy comparison plot saved to {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades: List[Dict],
                          title: str = "Trade Analysis",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detailed trade analysis.
        
        Args:
            trades (List[Dict]): List of trade dictionaries
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if not trades:
            logger.warning("No trades provided for analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract trade data
        profits = [trade.get('profit', 0) for trade in trades]
        durations = []
        
        for trade in trades:
            if 'buy_day' in trade and 'sell_day' in trade:
                duration = trade['sell_day'] - trade['buy_day']
                durations.append(duration)
        
        # Profit/Loss distribution
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        axes[0, 0].hist([winning_trades, losing_trades], bins=20, 
                       label=['Winning Trades', 'Losing Trades'],
                       color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('Profit/Loss Distribution')
        axes[0, 0].set_xlabel('Profit/Loss ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(profits)
        axes[0, 1].plot(cumulative_pnl, linewidth=2, color='blue')
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative Profit ($)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Trade duration analysis
        if durations:
            axes[1, 0].hist(durations, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Trade Duration Distribution')
            axes[1, 0].set_xlabel('Duration (Days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Profit by trade sequence
        axes[1, 1].bar(range(len(profits)), profits, 
                      color=['green' if p > 0 else 'red' for p in profits],
                      alpha=0.7)
        axes[1, 1].set_title('Profit by Trade Sequence')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Profit ($)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade analysis plot saved to {save_path}")
        
        return fig


class InteractivePlotter:
    """
    Interactive plotting using Plotly for web-based visualizations.
    """
    
    def __init__(self):
        pass
    
    def create_interactive_performance_chart(self, portfolio_values: List[float],
                                           benchmark_values: Optional[List[float]] = None,
                                           dates: Optional[List[str]] = None,
                                           title: str = "Interactive Portfolio Performance") -> go.Figure:
        """
        Create interactive performance chart using Plotly.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            benchmark_values (Optional[List[float]]): Benchmark values
            dates (Optional[List[str]]): Date labels
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        fig = go.Figure()
        
        x_axis = dates if dates else list(range(len(portfolio_values)))
        
        # Add portfolio trace
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=portfolio_values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add benchmark if provided
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=benchmark_values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date' if dates else 'Time Period',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_drawdown_chart(self, portfolio_values: List[float],
                                        dates: Optional[List[str]] = None,
                                        title: str = "Interactive Drawdown Analysis") -> go.Figure:
        """
        Create interactive drawdown chart.
        
        Args:
            portfolio_values (List[float]): Portfolio values
            dates (Optional[List[str]]): Date labels
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        # Calculate drawdowns
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak * 100
        
        x_axis = dates if dates else list(range(len(portfolio_values)))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value', 'Drawdown (%)'),
            vertical_spacing=0.1
        )
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # Peak values
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=peak,
            mode='lines',
            name='Peak Value',
            line=dict(color='red', width=1, dash='dash')
        ), row=1, col=1)
        
        # Drawdown
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=-drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red', width=0),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )   
        return fig