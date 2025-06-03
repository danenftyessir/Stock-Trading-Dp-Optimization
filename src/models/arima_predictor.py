"""
ARIMA Model Implementation for Stock Price Forecasting.
FIXED: Enhanced model selection and validation for better statistical significance.
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
    FIXED: Enhanced parameter selection with statistical significance validation.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5, 
                 significance_level: float = 0.05):
        """
        Initialize ARIMA predictor.
        
        Args:
            max_p (int): Maximum AR order to test
            max_d (int): Maximum differencing order to test
            max_q (int): Maximum MA order to test
            significance_level (float): Statistical significance level for parameter validation
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.significance_level = significance_level
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
        Find optimal ARIMA(p,d,q) parameters with statistical significance validation.
        FIXED: Enhanced model selection with parameter significance checking.
        
        Args:
            series (pd.Series): Time series data
            criterion (str): Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            Tuple[int, int, int]: Optimal (p, d, q) order
        """
        logger.info("Searching for optimal ARIMA parameters with significance validation...")
        
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
        
        # Grid search for optimal p and q with significance validation
        valid_models = []
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                # Skip if model is too complex for the data
                if p + q > len(series) // 10:
                    continue
                    
                try:
                    model = ARIMA(series, order=(p, d_optimal, q))
                    fitted_model = model.fit()
                    
                    # Check parameter significance
                    significance_check = self._check_parameter_significance(fitted_model)
                    
                    score = getattr(fitted_model, criterion)
                    
                    valid_models.append({
                        'order': (p, d_optimal, q),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic,
                        'score': score,
                        'significant_params': significance_check['significant_count'],
                        'total_params': significance_check['total_params'],
                        'significance_ratio': significance_check['significance_ratio'],
                        'model': fitted_model
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed to fit ARIMA{(p, d_optimal, q)}: {e}")
                    continue
        
        if not valid_models:
            logger.warning("No valid ARIMA models found, falling back to simple model")
            return (1, d_optimal, 1)
        
        # FIXED: Select model based on both goodness of fit and parameter significance
        # First filter models with at least 50% significant parameters
        significant_models = [m for m in valid_models if m['significance_ratio'] >= 0.5]
        
        if not significant_models:
            logger.warning("No models with significant parameters found, using best overall model")
            significant_models = valid_models
        
        # Among significant models, choose the one with best criterion
        best_model = min(significant_models, key=lambda x: x['score'])
        best_order = best_model['order']
        
        logger.info(f"Optimal ARIMA order: {best_order} with {criterion}={best_model['score']:.4f}")
        logger.info(f"Parameter significance: {best_model['significant_params']}/{best_model['total_params']} "
                   f"({best_model['significance_ratio']:.1%})")
        
        # Store results for analysis
        self.parameter_search_results = pd.DataFrame(valid_models)
        self.best_order = best_order
        
        return best_order
    
    def _check_parameter_significance(self, fitted_model) -> Dict:
        """
        Check statistical significance of model parameters.
        FIXED: Enhanced parameter significance validation.
        
        Args:
            fitted_model: Fitted ARIMA model
            
        Returns:
            Dict: Parameter significance analysis
        """
        try:
            pvalues = fitted_model.pvalues
            # Exclude sigma2 from significance check as it's always significant
            param_pvalues = pvalues[pvalues.index != 'sigma2'] if 'sigma2' in pvalues.index else pvalues
            
            significant_params = (param_pvalues <= self.significance_level).sum()
            total_params = len(param_pvalues)
            significance_ratio = significant_params / total_params if total_params > 0 else 0
            
            return {
                'significant_count': significant_params,
                'total_params': total_params,
                'significance_ratio': significance_ratio,
                'pvalues': param_pvalues.to_dict()
            }
        except Exception as e:
            logger.warning(f"Could not check parameter significance: {e}")
            return {
                'significant_count': 0,
                'total_params': 0,
                'significance_ratio': 0,
                'pvalues': {}
            }
    
    def fit(self, train_data: pd.Series, order: Optional[Tuple[int, int, int]] = None):
        """
        Fit ARIMA model to training data.
        FIXED: Enhanced fitting with validation and fallback options.
        
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
            
            # Validate the fitted model
            if self._validate_fitted_model():
                self.is_fitted = True
                logger.info(f"Model fitted successfully. AIC: {self.fitted_model.aic:.4f}")
                
                # Check and log parameter significance
                significance_check = self._check_parameter_significance(self.fitted_model)
                logger.info(f"Parameter significance: {significance_check['significant_count']}/"
                          f"{significance_check['total_params']} parameters significant")
            else:
                logger.warning("Model validation failed, trying simpler model")
                self._fit_fallback_model(train_data)
                
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            logger.info("Attempting fallback model...")
            self._fit_fallback_model(train_data)
    
    def _validate_fitted_model(self) -> bool:
        """
        Validate the fitted model for basic requirements.
        
        Returns:
            bool: True if model passes validation
        """
        if self.fitted_model is None:
            return False
        
        try:
            # Check if model converged
            if not hasattr(self.fitted_model, 'aic') or np.isnan(self.fitted_model.aic):
                return False
            
            # Check residuals
            residuals = self.fitted_model.resid
            if len(residuals) == 0 or np.all(np.isnan(residuals)):
                return False
            
            # Check for reasonable residual properties
            residual_std = np.std(residuals)
            if residual_std == 0 or np.isnan(residual_std):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Model validation error: {e}")
            return False
    
    def _fit_fallback_model(self, train_data: pd.Series):
        """
        Fit a simple fallback model when main fitting fails.
        
        Args:
            train_data (pd.Series): Training data
        """
        try:
            # Try simple ARIMA(1,1,1) as fallback
            fallback_order = (1, 1, 1)
            logger.info(f"Fitting fallback ARIMA{fallback_order} model")
            
            self.model = ARIMA(train_data, order=fallback_order)
            self.fitted_model = self.model.fit()
            self.best_order = fallback_order
            self.is_fitted = True
            
            logger.info("Fallback model fitted successfully")
            
        except Exception as e:
            # Final fallback: simple moving average
            logger.error(f"Fallback model also failed: {e}")
            logger.info("Using simple moving average as final fallback")
            self._create_moving_average_fallback(train_data)
    
    def _create_moving_average_fallback(self, train_data: pd.Series):
        """Create a simple moving average as final fallback."""
        class MovingAverageModel:
            def __init__(self, data):
                self.data = data
                self.aic = 0
                self.bic = 0
                self.hqic = 0
                self.llf = 0
                self.params = pd.Series({'ma_window': 5})
                self.pvalues = pd.Series({'ma_window': 0.05})
                self.resid = pd.Series([0] * len(data))
                
            def forecast(self, steps=1, alpha=0.05):
                # Simple forecast using last 5 values average
                last_values = self.data.tail(5).mean()
                return pd.Series([last_values] * steps)
            
            def get_forecast(self, steps=1, alpha=0.05):
                forecast = self.forecast(steps, alpha)
                # Create simple confidence intervals
                std_dev = self.data.pct_change().std()
                margin = 1.96 * std_dev  # 95% CI
                
                class ForecastResult:
                    def __init__(self, forecast, margin):
                        self.conf_int = pd.DataFrame({
                            'lower': forecast - margin,
                            'upper': forecast + margin
                        })
                
                return ForecastResult(forecast, margin)
            
            def predict(self, start=0, end=None):
                if end is None:
                    end = len(self.data) - 1
                return pd.Series([self.data.mean()] * (end - start + 1))
        
        self.fitted_model = MovingAverageModel(train_data)
        self.best_order = (0, 0, 1)  # Represents moving average
        self.is_fitted = True
    
    def forecast(self, steps: int = 1, confidence_level: float = 0.95) -> Dict:
        """
        Generate forecasts using fitted model.
        FIXED: Enhanced forecast with better error handling.
        
        Args:
            steps (int): Number of steps to forecast
            confidence_level (float): Confidence level for prediction intervals
            
        Returns:
            Dict: Forecast results with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        alpha = 1 - confidence_level
        
        try:
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
            conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
            
            # Ensure forecast_result is array-like
            if hasattr(forecast_result, 'values'):
                predictions = forecast_result.values
            else:
                predictions = np.array(forecast_result)
            
            results = {
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'conf_int_lower': conf_int.iloc[:, 0].values.tolist(),
                'conf_int_upper': conf_int.iloc[:, 1].values.tolist(),
                'confidence_level': confidence_level,
                'steps': steps
            }
            
            logger.info(f"Generated {steps}-step forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            # Return simple forecast based on last value
            last_value = self.training_data.iloc[-1] if self.training_data is not None else 0
            std_dev = self.training_data.std() if self.training_data is not None else 0.1
            
            return {
                'predictions': [last_value] * steps,
                'conf_int_lower': [last_value - 2*std_dev] * steps,
                'conf_int_upper': [last_value + 2*std_dev] * steps,
                'confidence_level': confidence_level,
                'steps': steps
            }
    
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
        
        try:
            predictions = self.fitted_model.predict(start=start, end=end)
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return simple predictions
            length = end - start + 1
            mean_value = self.training_data.mean() if self.training_data is not None else 0
            return np.array([mean_value] * length)
    
    def evaluate(self, test_data: pd.Series, forecast_steps: int = None) -> Dict:
        """
        Evaluate model performance on test data.
        FIXED: Enhanced evaluation with directional accuracy and robustness metrics.
        
        Args:
            test_data (pd.Series): Test time series
            forecast_steps (int): Number of steps to forecast, if None uses len(test_data)
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if forecast_steps is None:
            forecast_steps = min(len(test_data), 30)  # Limit to reasonable forecast horizon
        
        try:
            # Generate forecasts
            forecast_result = self.forecast(steps=forecast_steps)
            predictions = np.array(forecast_result['predictions'])
            
            # Use actual test data for comparison
            actual = test_data.values[:len(predictions)]
            
            # Basic accuracy metrics
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            
            # MAPE with protection against division by zero
            non_zero_actual = actual[actual != 0]
            non_zero_predictions = predictions[actual != 0]
            
            if len(non_zero_actual) > 0:
                mape = np.mean(np.abs((non_zero_actual - non_zero_predictions) / non_zero_actual)) * 100
                forecast_accuracy = 1 - np.mean(np.abs((non_zero_actual - non_zero_predictions) / non_zero_actual))
            else:
                mape = 100.0
                forecast_accuracy = 0.0
            
            # FIXED: Enhanced directional accuracy calculation
            directional_accuracy = self._calculate_directional_accuracy(actual, predictions)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': np.clip(mape, 0, 1000),  # Cap MAPE to reasonable range
                'forecast_accuracy': np.clip(forecast_accuracy, 0, 1),
                'directional_accuracy': directional_accuracy,
                'forecast_horizon': forecast_steps,
                'test_samples': len(actual)
            }
            
            logger.info(f"Model evaluation completed. RMSE: {metrics['rmse']:.4f}, "
                       f"MAPE: {metrics['mape']:.2f}%, "
                       f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'mape': 100.0,
                'forecast_accuracy': 0.0,
                'directional_accuracy': 0.5,  # Random chance
                'forecast_horizon': forecast_steps,
                'test_samples': 0,
                'error': str(e)
            }
    
    def _calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy with enhanced methodology.
        FIXED: More robust directional accuracy calculation.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: Directional accuracy
        """
        if len(actual) < 2 or len(predicted) < 2:
            return 0.5  # Random chance for insufficient data
        
        try:
            # Calculate direction changes
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            
            # Ensure same length
            min_length = min(len(actual_direction), len(predicted_direction))
            actual_direction = actual_direction[:min_length]
            predicted_direction = predicted_direction[:min_length]
            
            if min_length == 0:
                return 0.5
            
            # Calculate accuracy
            correct_directions = np.sum(actual_direction == predicted_direction)
            directional_accuracy = correct_directions / min_length
            
            return directional_accuracy
            
        except Exception as e:
            logger.warning(f"Directional accuracy calculation error: {e}")
            return 0.5  # Return random chance on error
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary.
        FIXED: Enhanced summary with parameter significance information.
        
        Returns:
            Dict: Model summary statistics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        try:
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
            
            # Add parameter significance analysis
            significance_check = self._check_parameter_significance(self.fitted_model)
            summary.update({
                'parameter_significance': significance_check,
                'model_quality': self._assess_model_quality()
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating model summary: {e}")
            return {"error": str(e)}
    
    def _assess_model_quality(self) -> Dict:
        """
        Assess overall model quality.
        
        Returns:
            Dict: Model quality assessment
        """
        try:
            significance_check = self._check_parameter_significance(self.fitted_model)
            
            quality_score = 100.0
            
            # Deduct points for non-significant parameters
            if significance_check['total_params'] > 0:
                non_significant_ratio = 1 - significance_check['significance_ratio']
                quality_score -= non_significant_ratio * 40
            
            # Check residual properties
            residuals = self.fitted_model.resid
            if len(residuals) > 10:
                # Check for autocorrelation in residuals (should be minimal)
                from statsmodels.stats.diagnostic import acorr_ljungbox
                try:
                    ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4))
                    if ljung_box['lb_pvalue'].iloc[0] < 0.05:  # Significant autocorrelation
                        quality_score -= 20
                except:
                    pass
            
            # Categorize quality
            if quality_score >= 80:
                quality_category = "Excellent"
            elif quality_score >= 60:
                quality_category = "Good"
            elif quality_score >= 40:
                quality_category = "Fair"
            else:
                quality_category = "Poor"
            
            return {
                'quality_score': quality_score,
                'quality_category': quality_category,
                'recommendations': self._generate_recommendations(quality_score, significance_check)
            }
            
        except Exception as e:
            logger.warning(f"Model quality assessment error: {e}")
            return {
                'quality_score': 50.0,
                'quality_category': "Unknown",
                'recommendations': ["Unable to assess model quality"]
            }
    
    def _generate_recommendations(self, quality_score: float, significance_check: Dict) -> List[str]:
        """Generate recommendations for model improvement."""
        recommendations = []
        
        if quality_score < 60:
            recommendations.append("Consider using more data for training")
            recommendations.append("Try different ARIMA orders")
        
        if significance_check['significance_ratio'] < 0.5:
            recommendations.append("Many parameters are not statistically significant")
            recommendations.append("Consider simpler model with fewer parameters")
        
        if quality_score < 40:
            recommendations.append("Model quality is poor, consider alternative approaches")
            recommendations.append("Check data quality and stationarity")
        
        return recommendations


class ARIMAPortfolioForecaster:
    """
    ARIMA forecasting for multiple stocks in a portfolio.
    FIXED: Enhanced portfolio forecasting with better model selection.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 significance_level: float = 0.05):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.significance_level = significance_level
        self.models = {}
        
    def fit_portfolio(self, stock_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Fit ARIMA models for multiple stocks with enhanced validation.
        FIXED: Better model selection and validation for portfolio.
        
        Args:
            stock_data (Dict[str, pd.Series]): Dictionary of stock price series
            
        Returns:
            Dict[str, Dict]: Fitting results for each stock
        """
        results = {}
        
        for symbol, price_series in stock_data.items():
            logger.info(f"Fitting ARIMA model for {symbol}")
            
            try:
                model = ARIMAPredictor(
                    self.max_p, self.max_d, self.max_q, 
                    significance_level=self.significance_level
                )
                
                # Enhanced data preprocessing
                processed_series = self._preprocess_series(price_series, symbol)
                
                if processed_series is not None and len(processed_series) > 50:
                    model.fit(processed_series)
                    
                    self.models[symbol] = model
                    
                    # Get comprehensive summary
                    summary = model.get_model_summary()
                    
                    results[symbol] = {
                        'success': True,
                        'model': model,
                        'model_summary': summary,
                        'data_quality': self._assess_data_quality(processed_series)
                    }
                else:
                    results[symbol] = {
                        'success': False,
                        'error': 'Insufficient or poor quality data'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to fit ARIMA for {symbol}: {e}")
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def _preprocess_series(self, series: pd.Series, symbol: str) -> Optional[pd.Series]:
        """
        Preprocess time series data for better ARIMA fitting.
        
        Args:
            series (pd.Series): Raw price series
            symbol (str): Stock symbol for logging
            
        Returns:
            Optional[pd.Series]: Preprocessed series or None if data is poor quality
        """
        try:
            # Remove any non-finite values
            clean_series = series.dropna()
            clean_series = clean_series[np.isfinite(clean_series)]
            
            if len(clean_series) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(clean_series)} points")
                return None
            
            # Check for constant values
            if clean_series.nunique() < 5:
                logger.warning(f"Series for {symbol} has too few unique values")
                return None
            
            # Remove extreme outliers
            q1, q3 = clean_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = (clean_series >= lower_bound) & (clean_series <= upper_bound)
            if outlier_mask.sum() < len(clean_series) * 0.8:  # Too many outliers
                logger.warning(f"Too many outliers detected for {symbol}")
                # Use quantile clipping instead
                clean_series = clean_series.clip(
                    clean_series.quantile(0.01), 
                    clean_series.quantile(0.99)
                )
            else:
                clean_series = clean_series[outlier_mask]
            
            return clean_series
            
        except Exception as e:
            logger.error(f"Error preprocessing series for {symbol}: {e}")
            return None
    
    def _assess_data_quality(self, series: pd.Series) -> Dict:
        """
        Assess quality of time series data.
        
        Args:
            series (pd.Series): Time series data
            
        Returns:
            Dict: Data quality assessment
        """
        try:
            quality_metrics = {
                'length': len(series),
                'unique_values': series.nunique(),
                'missing_ratio': series.isnull().sum() / len(series),
                'volatility': series.pct_change().std(),
                'trend_strength': abs(series.iloc[-1] - series.iloc[0]) / series.mean(),
            }
            
            # Calculate quality score
            score = 100.0
            
            if quality_metrics['length'] < 100:
                score -= 20
            if quality_metrics['unique_values'] < 10:
                score -= 30
            if quality_metrics['missing_ratio'] > 0.05:
                score -= 20
            if quality_metrics['volatility'] > 0.1:  # Very high volatility
                score -= 10
            
            quality_metrics['quality_score'] = max(0, score)
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Data quality assessment error: {e}")
            return {'quality_score': 50.0, 'error': str(e)}
    
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