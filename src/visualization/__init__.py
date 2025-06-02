"""
Visualization package for trading analysis charts and plots.
"""

# Import main plotting classes when available
try:
    from .performance_plots import PerformancePlotter, InteractivePlotter
    __all__ = ['PerformancePlotter', 'InteractivePlotter']
except ImportError:
    # Handle case where plotting dependencies might not be available
    __all__ = []