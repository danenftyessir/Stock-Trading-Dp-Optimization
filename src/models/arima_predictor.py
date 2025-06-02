"""
ARIMA Model Implementation for Stock Price Forecasting.
Exploratory ML integration as described in the research paper.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ARIMAPredictor:
    """
    ARIMA model for stock price forecasting.
    Supports automatic parameter selection and model evaluation.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        """
        Initialize ARIMA predictor.
        
        Args:
            max_p (int): Maximum AR order to test
            max_d (int): Maximum differencing order to test
            max_q (int): Maximum MA order to test
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.is_fitted = False
        
    def check_stationarity(self, timeseries: pd.Series, 
                          significance_level: float = 0.05) -> Dict:
        """
        Check stationarity using Augmented Dickey-Fuller test.
        
        Args:
            timeseries (pd.Series): Time series data
            significance_level (float): Significance level for ADF test
            
        Returns:
            Dict: Stationarity test results
        """
        result = adfuller(timeseries.dropna())
        
        stationarity_result = {
            'is_stationary': result[1] <= significance_level,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'significance_level': significance_level
        }
        
        logger.info(f"ADF Test: p-value = {result[1]:.6f}, "
                   f"Stationary = {stationarity_result['is_stationary']}")
        
        return stationarity_result
    
    def difference_series(self, series: pd.Series, order: int = 1) -> pd.Series:
        """
        Apply differencing to make series stationary.
        
        Args:
            series (pd.Series): Original time series
            order (int): Order of differencing
            
        Returns:
            pd.Series: Differenced series
        """
        differenced = series.copy()
        for _ in range(order):
            differenced = differenced.diff().dropna()
        return differenced
    
    def find_optimal_order(self, series: pd.Series, 
                          criterion: str = 'aic') -> Tuple[int, int, int]:
        """
        Find optimal ARIMA(p,d,q) parameters using grid search.
        
        Args:
            series (pd.Series): Time series data
            criterion (str): Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            Tuple[int, int, int]: Optimal (p, d, q) order
        """
        logger.info("Searching for optimal ARIMA parameters...")
        
        # Find optimal d (differencing order)
        d_optimal = 0
        series_test = series.copy()
        
        for d in range(self.max_d + 1):
            stationarity = self.check_stationarity(series_test)
            if stationarity['is_stationary']:
                d_optimal = d
                break
            if d < self.max_d:
                series_test = self.difference_series(series_test, 1)
        
        logger.info(f"Optimal d = {d_optimal}")
        
        # Grid search for optimal p and q
        best_score = float('inf')
        best_order = (0, d_optimal, 0)
        
        results = []
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d_optimal, q))
                    fitted_model = model.fit()
                    
                    score = getattr(fitted_model, criterion)
                    results.append({
                        'order': (p, d_optimal, q),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic,
                        'score': score
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d_optimal, q)
                        
                except Exception as e:
                    logger.debug(f"Failed to fit ARIMA{(p, d_optimal, q)}: {e}")
                    continue
        
        logger.info(f"Optimal ARIMA order: {best_order} with {criterion}={best_score:.4f}")
        
        # Store results for analysis
        self.parameter_search_results = pd.DataFrame(results)
        self.best_order = best_order
        
        return best_order
    
    def fit(self, train_data: pd.Series, order: Optional[Tuple[int, int, int]] = None):
        """
        Fit ARIMA model to training data.
        
        Args:
            train_data (pd.Series): Training time series
            order (Optional[Tuple[int, int, int]]): ARIMA order, if None will auto-select
        """
        self.training_data = train_data.copy()
        
        if order is None:
            order = self.find_optimal_order(train_data)
        
        self.best_order = order
        
        logger.info(f"Fitting ARIMA{order} model...")
        
        try:
            self.model = ARIMA(train_data, order=order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            logger.info(f"Model fitted successfully. AIC: {self.fitted_model.aic:.4f}")
            logger.info(f"Model summary:\n{self.fitted_model.summary()}")
            
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise
    
    def forecast(self, steps: int = 1, confidence_level: float = 0.95) -> Dict:
        """
        Generate forecasts using fitted model.
        
        Args:
            steps (int): Number of steps to forecast
            confidence_level (float): Confidence level for prediction intervals
            
        Returns:
            Dict: Forecast results with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        alpha = 1 - confidence_level
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
        conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
        
        results = {
            'predictions': forecast_result.values,
            'conf_int_lower': conf_int.iloc[:, 0].values,
            'conf_int_upper': conf_int.iloc[:, 1].values,
            'confidence_level': confidence_level,
            'steps': steps
        }
        
        logger.info(f"Generated {steps}-step forecast")
        
        return results
    
    def predict(self, start: int, end: int) -> np.ndarray:
        """
        Generate in-sample or out-of-sample predictions.
        
        Args:
            start (int): Start index for prediction
            end (int): End index for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model.predict(start=start, end=end)
    
    def evaluate(self, test_data: pd.Series, forecast_steps: int = None) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.Series): Test time series
            forecast_steps (int): Number of steps to forecast, if None uses len(test_data)
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if forecast_steps is None:
            forecast_steps = len(test_data)
        
        # Generate forecasts
        forecast_result = self.forecast(steps=forecast_steps)
        predictions = forecast_result['predictions']
        
        # Calculate metrics
        actual = test_data.values[:len(predictions)]
        
        metrics = {
            'mse': mean_squared_error(actual, predictions),
            'rmse': np.sqrt(mean_squared_error(actual, predictions)),
            'mae': mean_absolute_error(actual, predictions),
            'mape': np.mean(np.abs((actual - predictions) / actual)) * 100,
            'forecast_accuracy': 1 - np.mean(np.abs((actual - predictions) / actual)),
            'directional_accuracy': self._directional_accuracy(actual, predictions)
        }
        
        logger.info(f"Model evaluation completed. RMSE: {metrics['rmse']:.4f}, "
                   f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def _directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy (correct prediction of up/down movement).
        """
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
            
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        
        return np.mean(actual_direction == predicted_direction)
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot model diagnostics including residuals and ACF/PACF.
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals plot
        residuals = self.fitted_model.resid
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 0], lags=40, alpha=0.05)
        axes[1, 0].set_title('ACF of Residuals')
        
        # PACF of residuals
        plot_pacf(residuals, ax=axes[1, 1], lags=40, alpha=0.05)
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        return fig
    
    def plot_forecast(self, test_data: pd.Series = None, 
                     forecast_steps: int = 30, 
                     figsize: Tuple[int, int] = (12, 6)):
        """
        Plot historical data, fitted values, and forecasts.
        
        Args:
            test_data (pd.Series): Test data to compare with forecasts
            forecast_steps (int): Number of steps to forecast
            figsize (Tuple[int, int]): Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecast")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data
        train_idx = range(len(self.training_data))
        ax.plot(train_idx, self.training_data.values, label='Training Data', color='blue')
        
        # Plot fitted values
        fitted_values = self.fitted_model.fittedvalues
        ax.plot(train_idx, fitted_values, label='Fitted Values', color='red', alpha=0.7)
        
        # Generate and plot forecast
        forecast_result = self.forecast(steps=forecast_steps)
        forecast_idx = range(len(self.training_data), 
                           len(self.training_data) + forecast_steps)
        
        ax.plot(forecast_idx, forecast_result['predictions'], 
               label='Forecast', color='green', linestyle='--')
        
        # Plot confidence intervals
        ax.fill_between(forecast_idx, 
                       forecast_result['conf_int_lower'],
                       forecast_result['conf_int_upper'],
                       alpha=0.3, color='green', label='Confidence Interval')
        
        # Plot test data if provided
        if test_data is not None:
            test_idx = range(len(self.training_data), 
                           len(self.training_data) + len(test_data))
            ax.plot(test_idx, test_data.values, label='Actual Test Data', 
                   color='orange', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(f'ARIMA{self.best_order} Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary.
        
        Returns:
            Dict: Model summary statistics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        summary = {
            'order': self.best_order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'log_likelihood': self.fitted_model.llf,
            'params': self.fitted_model.params.to_dict(),
            'pvalues': self.fitted_model.pvalues.to_dict(),
            'training_samples': len(self.training_data),
            'residual_std': self.fitted_model.resid.std()
        }
        
        return summary


class ARIMAPortfolioForecaster:
    """
    ARIMA forecasting for multiple stocks in a portfolio.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.models = {}
        
    def fit_portfolio(self, stock_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Fit ARIMA models for multiple stocks.
        
        Args:
            stock_data (Dict[str, pd.Series]): Dictionary of stock price series
            
        Returns:
            Dict[str, Dict]: Fitting results for each stock
        """
        results = {}
        
        for symbol, price_series in stock_data.items():
            logger.info(f"Fitting ARIMA model for {symbol}")
            
            try:
                model = ARIMAPredictor(self.max_p, self.max_d, self.max_q)
                model.fit(price_series)
                
                self.models[symbol] = model
                results[symbol] = {
                    'success': True,
                    'model': model,
                    'summary': model.get_model_summary()
                }
                
            except Exception as e:
                logger.error(f"Failed to fit ARIMA for {symbol}: {e}")
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def forecast_portfolio(self, steps: int = 5) -> Dict[str, Dict]:
        """
        Generate forecasts for all fitted models.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, Dict]: Forecast results for each stock
        """
        forecasts = {}
        
        for symbol, model in self.models.items():
            try:
                forecast_result = model.forecast(steps=steps)
                forecasts[symbol] = forecast_result
                
            except Exception as e:
                logger.error(f"Failed to forecast {symbol}: {e}")
                forecasts[symbol] = {'error': str(e)}
        
        return forecasts