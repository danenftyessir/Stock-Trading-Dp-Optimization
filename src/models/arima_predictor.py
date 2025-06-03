"""
ARIMA Model Implementation for Stock Price Forecasting.
FIXED: Enhanced model selection and validation for better statistical significance.
REVISED: Improved model quality assessment and recommendations based on user feedback.
COMPLETED: Finalized ARIMAPortfolioForecaster and removed dummy/hardcoded values.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
# Removed unused acf, pacf, plot_acf, plot_pacf to clean up imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Removed unused matplotlib and seaborn imports as plotting is not done in this module
from typing import Tuple, List, Dict, Optional
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ARIMAPredictor:
    """
    ARIMA model for stock price forecasting.
    REVISED: Improved model quality assessment and recommendations.
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
        self.best_order: Optional[Tuple[int, int, int]] = None
        self.model: Optional[ARIMA] = None
        self.fitted_model: Optional[any] = None # statsmodels.tsa.statespace.mlemodel.MLEResultsWrapper
        self.training_data: Optional[pd.Series] = None
        self.is_fitted: bool = False
        self.evaluation_metrics: Dict[str, any] = {} # To store evaluation metrics
        self.parameter_search_results: Optional[pd.DataFrame] = None

        
    def check_stationarity(self, timeseries: pd.Series, 
                          significance_level: float = 0.05) -> Dict[str, any]:
        """
        Check stationarity using Augmented Dickey-Fuller test.
        
        Args:
            timeseries (pd.Series): Time series data
            significance_level (float): Significance level for ADF test
            
        Returns:
            Dict: Stationarity test results
        """
        # Drop NA values before ADF test, as it cannot handle them
        cleaned_timeseries = timeseries.dropna()
        if cleaned_timeseries.empty:
            logger.warning("Timeseries is empty after dropping NA values. Cannot perform ADF test.")
            return {
                'is_stationary': False, # Default to non-stationary if no data
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'significance_level': significance_level,
                'error': 'Empty series after NA drop'
            }

        result = adfuller(cleaned_timeseries)
        
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
        REVISED: Prioritize models with better parameter significance if AIC/BIC scores are close.
        
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
        
        for d_val in range(self.max_d + 1): 
            stationarity = self.check_stationarity(series_test)
            if stationarity.get('is_stationary', False): # Check for 'is_stationary' key
                d_optimal = d_val
                break
            if d_val < self.max_d:
                if series_test.empty: # if differencing results in empty series
                    logger.warning(f"Series became empty after differencing {d_val+1} time(s). Stopping d search.")
                    break
                series_test = self.difference_series(series_test, 1)
        
        logger.info(f"Optimal d = {d_optimal}")
        
        valid_models = []
        
        for p_val in range(self.max_p + 1): 
            for q_val in range(self.max_q + 1): 
                # Skip if model is too complex for the data, but allow (0,d,0)
                if p_val + q_val > 0 and p_val + q_val > len(series) // 10 : 
                    continue
                    
                try:
                    current_order = (p_val, d_optimal, q_val)
                    # Ensure series has enough observations for the order
                    if len(series) < sum(current_order) + 10: # Heuristic: need some buffer
                        logger.debug(f"Skipping ARIMA{current_order}: not enough data points ({len(series)})")
                        continue

                    model = ARIMA(series, order=current_order)
                    fitted_model = model.fit()
                    
                    significance_check = self._check_parameter_significance(fitted_model, current_order)
                    
                    score = getattr(fitted_model, criterion, np.inf) # Default to inf if criterion not found
                    
                    valid_models.append({
                        'order': current_order,
                        'aic': getattr(fitted_model, 'aic', np.inf),
                        'bic': getattr(fitted_model, 'bic', np.inf),
                        'hqic': getattr(fitted_model, 'hqic', np.inf),
                        'score': score,
                        'significant_params': significance_check['significant_count'],
                        'total_params': significance_check['total_params'],
                        'significance_ratio': significance_check['significance_ratio'],
                        'model_object': fitted_model # Store the fitted model object for potential reuse
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed to fit ARIMA{current_order}: {e}")
                    continue
        
        if not valid_models:
            logger.warning("No valid ARIMA models found during grid search.")
            # Fallback logic: try to fit the simplest model (0,d,0) or (1,d,0) etc.
            for fallback_p, fallback_q in [(0,0), (1,0), (0,1), (1,1)]:
                try:
                    fallback_order = (fallback_p, d_optimal, fallback_q)
                    ARIMA(series, order=fallback_order).fit()
                    self.best_order = fallback_order
                    logger.info(f"Using fallback order: {self.best_order}")
                    return self.best_order
                except:
                    continue
            logger.error("All fallback models failed. Data might be unsuitable for ARIMA.")
            self.best_order = (0, d_optimal, 0) # Default to a very simple model
            return self.best_order

        valid_models_sorted = sorted(
            valid_models, 
            key=lambda x: (x['score'], -x['significance_ratio'], x['total_params'])
        )
        
        best_model_info = valid_models_sorted[0]
        best_order = best_model_info['order']
        
        logger.info(f"Optimal ARIMA order: {best_order} with {criterion}={best_model_info['score']:.4f}")
        logger.info(f"Parameter significance: {best_model_info['significant_params']}/{best_model_info['total_params']} "
                   f"({best_model_info['significance_ratio']:.1%})")
        
        self.parameter_search_results = pd.DataFrame([
            {k: v for k, v in model_entry.items() if k != 'model_object'} for model_entry in valid_models
        ])
        self.best_order = best_order
        # Optionally store the best fitted model object from search to avoid re-fitting if needed
        # self.fitted_model = best_model_info.get('model_object') 
        # self.is_fitted = True if self.fitted_model else False
        
        return best_order
    
    def _check_parameter_significance(self, fitted_model, order: Tuple[int,int,int]) -> Dict[str, any]:
        """
        Check statistical significance of model parameters.
        REVISED: Correctly identify AR/MA parameters.
        
        Args:
            fitted_model: Fitted ARIMA model
            order: The (p,d,q) order of the model
            
        Returns:
            Dict: Parameter significance analysis
        """
        p_val, _, q_val = order
        results: Dict[str, any] = { # Initialize with default values
            'significant_count': 0,
            'total_params': 0,
            'significance_ratio': 1.0, 
            'pvalues': {}
        }
        try:
            pvalues = fitted_model.pvalues
            
            ar_params_names = [f'ar.L{i}' for i in range(1, p_val + 1) if f'ar.L{i}' in pvalues.index]
            ma_params_names = [f'ma.L{i}' for i in range(1, q_val + 1) if f'ma.L{i}' in pvalues.index]
            
            relevant_param_names = ar_params_names + ma_params_names
            
            if not relevant_param_names: 
                results['total_params'] = 0 # Explicitly set for (0,d,0)
                return results

            param_pvalues = pvalues.loc[relevant_param_names]
            
            results['significant_count'] = int((param_pvalues <= self.significance_level).sum())
            results['total_params'] = len(param_pvalues)
            results['significance_ratio'] = results['significant_count'] / results['total_params'] if results['total_params'] > 0 else 1.0
            results['pvalues'] = param_pvalues.to_dict()
            
            return results
        except Exception as e:
            logger.warning(f"Could not check parameter significance for order {order}: {e}")
            results['total_params'] = p_val + q_val # Expected number
            results['significance_ratio'] = 0.0 # Assume none are significant if check fails
            return results
    
    def fit(self, train_data: pd.Series, order: Optional[Tuple[int, int, int]] = None):
        """
        Fit ARIMA model to training data.
        
        Args:
            train_data (pd.Series): Training time series
            order (Optional[Tuple[int, int, int]]): ARIMA order, if None will auto-select
        """
        if train_data.empty:
            logger.error("Training data is empty. Cannot fit ARIMA model.")
            self.is_fitted = False
            return

        self.training_data = train_data.copy()
        
        if order is None:
            # Check if series is constant, which find_optimal_order might struggle with
            if self.training_data.nunique() == 1:
                logger.warning("Training data is constant. Using ARIMA(0,0,0) or (0,1,0).")
                # Try (0,1,0) first if data is long enough for differencing
                d_order = 1 if len(self.training_data) > 1 else 0
                order = (0, d_order, 0)
            else:
                order = self.find_optimal_order(self.training_data)
        
        self.best_order = order
        
        logger.info(f"Fitting ARIMA{order} model...")
        
        try:
            self.model = ARIMA(self.training_data, order=order)
            self.fitted_model = self.model.fit()
            
            if self._validate_fitted_model():
                self.is_fitted = True
                logger.info(f"Model fitted successfully. AIC: {getattr(self.fitted_model, 'aic', 'N/A'):.4f}")
                significance_check = self._check_parameter_significance(self.fitted_model, self.best_order)
                logger.info(f"Parameter significance: {significance_check['significant_count']}/"
                          f"{significance_check['total_params']} AR/MA parameters significant.")
            else:
                logger.warning("Model validation failed after fitting. Attempting fallback.")
                self._fit_fallback_model(self.training_data)
                
        except Exception as e:
            logger.error(f"Failed to fit ARIMA{order} model: {e}")
            logger.info("Attempting fallback model...")
            self._fit_fallback_model(self.training_data)
    
    def _validate_fitted_model(self) -> bool:
        """
        Validate the fitted model for basic requirements.
        """
        if self.fitted_model is None:
            logger.debug("Validation failed: fitted_model is None.")
            return False
        
        try:
            if not hasattr(self.fitted_model, 'aic') or pd.isna(self.fitted_model.aic):
                logger.warning("Validation failed: Model AIC is NaN or missing.")
                return False
            
            residuals = self.fitted_model.resid
            if residuals.empty or residuals.isna().all():
                logger.warning("Validation failed: Model residuals are empty or all NaN.")
                return False
            
            residual_std = np.std(residuals)
            if pd.isna(residual_std) or residual_std == 0 :
                # Allow near-zero std for constant series after differencing, e.g. (0,1,0) on a ramp
                if self.training_data is not None and self.training_data.nunique() > 1 and self.best_order != (0,1,0):
                    logger.warning(f"Validation failed: Model residuals have zero or NaN standard deviation ({residual_std}).")
                    return False
                elif self.best_order == (0,1,0) and self.training_data is not None and self.training_data.nunique() == 1:
                    logger.debug("Residual std is zero for ARIMA(0,1,0) on constant series, this is expected.")
                elif residual_std == 0 and self.training_data is not None and self.training_data.nunique() > 1 :
                     logger.warning(f"Validation failed: Model residuals have zero standard deviation ({residual_std}) for non-constant data.")
                     return False


            return True
            
        except Exception as e:
            logger.warning(f"Model validation error: {e}")
            return False
    
    def _fit_fallback_model(self, train_data: pd.Series):
        """
        Fit a simple fallback model when main fitting fails.
        """
        try:
            d_optimal = self.best_order[1] if self.best_order and self.best_order[1] <= self.max_d else 0
            # Try (0,d,0) first
            fallback_order_0d0 = (0, d_optimal, 0)
            logger.info(f"Attempting fallback ARIMA{fallback_order_0d0} model")
            self.model = ARIMA(train_data, order=fallback_order_0d0)
            self.fitted_model = self.model.fit()
            if self._validate_fitted_model():
                self.best_order = fallback_order_0d0
                self.is_fitted = True
                logger.info(f"Fallback ARIMA{fallback_order_0d0} model fitted successfully.")
                return
            
            # Try (1,d,0) if (0,d,0) failed validation
            fallback_order_1d0 = (1, d_optimal, 0)
            logger.info(f"Attempting fallback ARIMA{fallback_order_1d0} model")
            self.model = ARIMA(train_data, order=fallback_order_1d0)
            self.fitted_model = self.model.fit()
            if self._validate_fitted_model():
                self.best_order = fallback_order_1d0
                self.is_fitted = True
                logger.info(f"Fallback ARIMA{fallback_order_1d0} model fitted successfully.")
                return

        except Exception as e_fallback:
            logger.error(f"All ARIMA fallback models also failed: {e_fallback}")
            self._create_moving_average_fallback(train_data) # Use the placeholder MA
    
    def _create_moving_average_fallback(self, train_data: pd.Series, window: int = 5):
        """Create a simple moving average as final fallback if ARIMA fails."""
        class MovingAverageModel: # Simplified placeholder
            def __init__(self, data: pd.Series, window_size: int):
                self.data = data
                self.window = window_size
                self.aic = np.inf 
                self.bic = np.inf
                self.hqic = np.inf
                self.llf = -np.inf
                self.params = pd.Series({'ma_window': self.window, 'sigma2': data.var(ddof=0) if len(data)>0 else 0.0})
                self.pvalues = pd.Series({'ma_window': 0.0, 'sigma2': 0.0}) 
                # Simplified residuals: difference from rolling mean
                if not data.empty:
                    rolling_mean = data.rolling(window=self.window, min_periods=1).mean()
                    self.resid = (data - rolling_mean).fillna(0)
                else:
                    self.resid = pd.Series(dtype=float)


            def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.Series:
                if self.data.empty: return pd.Series([0.0] * steps)
                last_val_mean = self.data.tail(self.window).mean() if len(self.data) >= self.window else self.data.mean()
                return pd.Series([last_val_mean if pd.notna(last_val_mean) else 0.0] * steps)

            def get_forecast(self, steps: int = 1, alpha: float = 0.05):
                forecast_series = self.forecast(steps=steps, alpha=alpha)
                std_dev = self.data.std() if not self.data.empty else 0.1
                margin = 1.96 * std_dev # Simplified margin
                
                conf_int_df = pd.DataFrame({
                    'lower': forecast_series - margin,
                    'upper': forecast_series + margin
                })
                class ForecastResultWrapper:
                    def __init__(self, pred_mean, conf_int_df_):
                        self.predicted_mean = pred_mean
                        self._conf_int = conf_int_df_
                    def conf_int(self): 
                        return self._conf_int
                return ForecastResultWrapper(forecast_series, conf_int_df)

            def predict(self, start: int = 0, end: Optional[int] = None, dynamic: bool = False) -> pd.Series:
                if self.data.empty: return pd.Series(dtype=float)
                if end is None: end = len(self.data) - 1
                
                # For in-sample, use rolling mean; for out-of-sample, use last known mean
                if start < len(self.data):
                    predictions = self.data.rolling(window=self.window, min_periods=1).mean().fillna(method='bfill')
                    # Ensure predictions cover the requested range, extending with last known mean if needed
                    if end >= len(self.data):
                        last_known_mean = predictions.iloc[-1] if not predictions.empty else self.data.mean()
                        extended_preds = pd.Series([last_known_mean] * (end - len(self.data) + 1))
                        predictions = pd.concat([predictions, extended_preds], ignore_index=True)
                    return predictions[start : end + 1]
                else: # Pure out-of-sample prediction
                    last_val_mean = self.data.tail(self.window).mean() if len(self.data) >= self.window else self.data.mean()
                    return pd.Series([last_val_mean if pd.notna(last_val_mean) else 0.0] * (end - start + 1))

        self.fitted_model = MovingAverageModel(train_data, window=window)
        self.best_order = (0,0,0) # Representing a non-ARIMA fallback
        self.is_fitted = True
        logger.info(f"Created a simple Moving Average (window={window}) fallback model as ARIMA fitting failed.")


    def forecast(self, steps: int = 1, confidence_level: float = 0.95) -> Dict[str, any]:
        """
        Generate forecasts using fitted model.
        """
        if not self.is_fitted or self.fitted_model is None:
            logger.error("Model must be fitted before forecasting.")
            last_value = 0.0
            std_dev = 0.1
            if self.training_data is not None and not self.training_data.empty:
                last_value = self.training_data.iloc[-1]
                std_dev = self.training_data.std() if self.training_data.std() else 0.1

            return {
                'predictions': [last_value] * steps,
                'conf_int_lower': [last_value - 2 * std_dev] * steps,
                'conf_int_upper': [last_value + 2 * std_dev] * steps,
                'confidence_level': confidence_level,
                'steps': steps,
                'error': "Model not fitted or fitting failed."
            }

        alpha = 1 - confidence_level
        
        try:
            forecast_obj = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
            predictions = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            
            predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

            results = {
                'predictions': predictions_list,
                'conf_int_lower': conf_int.iloc[:, 0].values.tolist(),
                'conf_int_upper': conf_int.iloc[:, 1].values.tolist(),
                'confidence_level': confidence_level,
                'steps': steps
            }
            logger.info(f"Generated {steps}-step forecast")
            return results
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            last_value = self.training_data.iloc[-1] if self.training_data is not None and not self.training_data.empty else 0.0
            std_dev = self.training_data.std() if self.training_data is not None and self.training_data.std() else 0.1
            
            return {
                'predictions': [last_value] * steps,
                'conf_int_lower': [last_value - 2 * std_dev] * steps,
                'conf_int_upper': [last_value + 2 * std_dev] * steps,
                'confidence_level': confidence_level,
                'steps': steps,
                'error': str(e)
            }
    
    def predict(self, start: int, end: int) -> np.ndarray:
        """
        Generate in-sample or out-of-sample predictions.
        """
        if not self.is_fitted or self.fitted_model is None:
            logger.error("Model must be fitted before prediction.")
            length = end - start + 1
            mean_value = 0.0
            if self.training_data is not None and not self.training_data.empty:
                mean_value = self.training_data.mean()
            return np.array([mean_value if pd.notna(mean_value) else 0.0] * length)

        try:
            # Ensure start and end are within reasonable bounds for the model
            # For statsmodels ARIMA, predict can go beyond len(original_series) for out-of-sample
            predictions = self.fitted_model.predict(start=start, end=end)
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            length = end - start + 1
            mean_value = self.training_data.mean() if self.training_data is not None and not self.training_data.empty else 0.0
            return np.array([mean_value if pd.notna(mean_value) else 0.0] * length)
    
    def evaluate(self, test_data: pd.Series, forecast_steps: Optional[int] = None) -> Dict[str, any]:
        """
        Evaluate model performance on test data.
        """
        self.evaluation_metrics = {} # Reset evaluation metrics

        if not self.is_fitted or self.fitted_model is None:
            logger.error("Model must be fitted before evaluation.")
            self.evaluation_metrics.update({
                'error': "Model not fitted", 'directional_accuracy': 0.5,
                'mse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'mape': 100.0, 'forecast_accuracy': 0.0,
                'forecast_horizon': 0, 'test_samples': 0
            })
            return self.evaluation_metrics.copy()
        
        if test_data.empty:
            logger.warning("Test data is empty. Cannot evaluate.")
            self.evaluation_metrics.update({
                'error': "Empty test data", 'directional_accuracy': 0.5,
                'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'forecast_accuracy': np.nan,
                'forecast_horizon': 0, 'test_samples': 0
            })
            return self.evaluation_metrics.copy()

        if forecast_steps is None:
            forecast_steps = len(test_data) 
        forecast_steps = min(forecast_steps, len(test_data)) # Cannot forecast more steps than available test data
        
        if forecast_steps <= 0:
            logger.warning("Forecast steps is non-positive. Cannot evaluate.")
            self.evaluation_metrics.update({
                'error': "Non-positive forecast steps", 'directional_accuracy': 0.5,
                'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'forecast_accuracy': np.nan,
                'forecast_horizon': 0, 'test_samples': 0
            })
            return self.evaluation_metrics.copy()

        try:
            forecast_result = self.forecast(steps=forecast_steps)
            if 'error' in forecast_result:
                logger.error(f"Forecasting failed during evaluation: {forecast_result['error']}")
                self.evaluation_metrics.update(forecast_result) # Include error from forecast
                self.evaluation_metrics.setdefault('directional_accuracy', 0.5)
                return self.evaluation_metrics.copy()

            predictions = np.array(forecast_result['predictions'])
            actual = test_data.values[:len(predictions)]

            if len(actual) == 0:
                 logger.warning("Aligned actual test data is empty. Cannot evaluate.")
                 self.evaluation_metrics.update({'error': "Aligned test data empty", 'directional_accuracy': 0.5})
                 return self.evaluation_metrics.copy()
            
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            
            non_zero_actual_mask = actual != 0
            non_zero_actual = actual[non_zero_actual_mask]
            non_zero_predictions = predictions[non_zero_actual_mask]
            
            mape = np.nan
            forecast_accuracy = np.nan
            if len(non_zero_actual) > 0:
                mape_values = np.abs((non_zero_actual - non_zero_predictions) / non_zero_actual)
                mape = np.mean(mape_values) * 100
                forecast_accuracy = 1 - np.mean(mape_values)
            elif len(actual) > 0 : # All actuals are zero
                mape = 100.0 if np.any(predictions != 0) else 0.0 # MAPE is 100% if predictions are non-zero
                forecast_accuracy = 0.0 if np.any(predictions != 0) else 1.0


            directional_accuracy = self._calculate_directional_accuracy(actual, predictions)
            
            self.evaluation_metrics.update({
                'mse': mse, 'rmse': rmse, 'mae': mae,
                'mape': np.clip(mape, 0, 1000) if pd.notna(mape) else np.nan,
                'forecast_accuracy': np.clip(forecast_accuracy, 0, 1) if pd.notna(forecast_accuracy) else np.nan,
                'directional_accuracy': directional_accuracy,
                'forecast_horizon': forecast_steps,
                'test_samples': len(actual)
            })
            
            logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAPE={mape:.2f}%, DirAcc={directional_accuracy:.2%}")
            return self.evaluation_metrics.copy()
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            self.evaluation_metrics.update({
                'error': str(e), 'directional_accuracy': 0.5,
                'mse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'mape': 100.0, 'forecast_accuracy': 0.0,
                'forecast_horizon': forecast_steps, 'test_samples': 0
            })
            return self.evaluation_metrics.copy()
    
    def _calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy.
        """
        if len(actual) < 2 or len(predicted) < 2: return 0.5
        
        try:
            actual_diff = np.diff(actual)
            predicted_diff = np.diff(predicted)
            min_len = min(len(actual_diff), len(predicted_diff))
            if min_len == 0: return 0.5

            actual_direction = actual_diff[:min_len] > 0
            predicted_direction = predicted_diff[:min_len] > 0
            
            correct_directions = np.sum(actual_direction == predicted_direction)
            return correct_directions / min_len
        except: return 0.5
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get comprehensive model summary.
        """
        if not self.is_fitted or self.fitted_model is None:
            return {"error": "Model not fitted"}
        
        try:
            summary = {
                'order': self.best_order,
                'aic': getattr(self.fitted_model, 'aic', np.nan),
                'bic': getattr(self.fitted_model, 'bic', np.nan),
                'hqic': getattr(self.fitted_model, 'hqic', np.nan),
                'log_likelihood': getattr(self.fitted_model, 'llf', np.nan),
                'params': getattr(self.fitted_model, 'params', pd.Series(dtype=float)).to_dict(),
                'pvalues': getattr(self.fitted_model, 'pvalues', pd.Series(dtype=float)).to_dict(),
                'training_samples': len(self.training_data) if self.training_data is not None else 0,
            }
            if hasattr(self.fitted_model, 'resid') and not self.fitted_model.resid.empty:
                summary['residual_std'] = self.fitted_model.resid.std()
            else:
                summary['residual_std'] = np.nan

            significance_check = self._check_parameter_significance(self.fitted_model, self.best_order)
            current_eval_metrics = self.evaluation_metrics.copy() # Use stored eval metrics

            summary.update({
                'parameter_significance': significance_check,
                'model_quality': self._assess_model_quality(significance_check, current_eval_metrics) 
            })
            
            if current_eval_metrics and 'error' not in current_eval_metrics:
                summary['evaluation_metrics'] = current_eval_metrics

            return summary
        except Exception as e:
            logger.error(f"Error generating model summary: {e}")
            return {"error": str(e)}
    
    def _assess_model_quality(self, significance_check: Dict, eval_metrics: Optional[Dict] = None) -> Dict[str, any]:
        """
        Assess overall model quality.
        """
        eval_metrics = eval_metrics if eval_metrics is not None else {}
        quality_score = 100.0
        
        p_val, d_val, q_val = self.best_order if self.best_order else (0,0,0)
        is_random_walk_equivalent = (p_val == 0 and q_val == 0)

        recommendations = self._generate_recommendations(significance_check, eval_metrics, self.best_order)

        if is_random_walk_equivalent:
            quality_score = 50.0 
        else:
            if significance_check.get('total_params', 0) > 0:
                non_significant_ratio = 1 - significance_check.get('significance_ratio', 0.0)
                quality_score -= non_significant_ratio * 40
            elif p_val > 0 or q_val > 0: 
                 quality_score -= 50

        directional_accuracy = eval_metrics.get('directional_accuracy')
        if pd.notna(directional_accuracy):
            if directional_accuracy < 0.5: quality_score -= 30
            elif directional_accuracy < 0.55: quality_score -= 15
        elif eval_metrics and 'error' not in eval_metrics:
             quality_score -= 10 

        if hasattr(self.fitted_model, 'resid') and not self.fitted_model.resid.empty and len(self.fitted_model.resid) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lags_lb = min(10, max(1, len(self.fitted_model.resid)//5)) # Ensure lags_lb is at least 1
                ljung_box_result = acorr_ljungbox(self.fitted_model.resid.dropna(), lags=[lags_lb], return_df=True)
                if not ljung_box_result.empty and ljung_box_result['lb_pvalue'].iloc[-1] < 0.05:
                    quality_score -= 20
                    logger.debug(f"Residuals show autocorrelation (Ljung-Box p-value: {ljung_box_result['lb_pvalue'].iloc[-1]:.3f})")
            except Exception as e_res: logger.warning(f"Residual check error: {e_res}")
        
        quality_score = max(0, quality_score)
        quality_category = "Poor" # Default
        if is_random_walk_equivalent:
            quality_category = "Fair (Random Walk)" if quality_score >= 45 else "Poor (Random Walk)"
        elif pd.notna(directional_accuracy) and directional_accuracy < 0.5:
            quality_category = "Poor (Low Directional Accuracy)"
        elif quality_score >= 85: quality_category = "Excellent"
        elif quality_score >= 70: quality_category = "Good"
        elif quality_score >= 50: quality_category = "Fair"
            
        return {
            'quality_score': quality_score,
            'quality_category': quality_category,
            'recommendations': recommendations
        }

    def _generate_recommendations(self, significance_check: Dict, eval_metrics: Optional[Dict], order: Optional[Tuple[int,int,int]]) -> List[str]:
        """
        Generate recommendations for model improvement.
        """
        recommendations = []
        eval_metrics = eval_metrics if eval_metrics is not None else {}
        p_val, _, q_val = order if order else (0,0,0) # Unpack d_val as well
        is_random_walk_equivalent = (p_val == 0 and q_val == 0)

        if is_random_walk_equivalent:
            recommendations.append(
                "Model is ARIMA(0,d,0) (Random Walk on differenced series). "
                "Predictive power for complex patterns is limited."
            )
            if eval_metrics.get('directional_accuracy', 0.5) <= 0.5:
                 recommendations.append("Directional accuracy is at/below random chance, as expected for this model type.")
            recommendations.append("Consider if this simple model is adequate or if alternative forecasting approaches are needed.")
        else: 
            if significance_check.get('significance_ratio', 1.0) < 0.5 and significance_check.get('total_params',0) > 0:
                recommendations.append(
                    "Many AR/MA parameters are not statistically significant. "
                    "Consider reducing p or q, or re-evaluating model selection."
                )
            elif significance_check.get('total_params',0) == 0 and (p_val > 0 or q_val > 0):
                 recommendations.append("Model has AR/MA terms, but no parameters were found/checked for significance. Review fitting process.")

        directional_accuracy = eval_metrics.get('directional_accuracy')
        if pd.notna(directional_accuracy):
            if directional_accuracy < 0.5:
                recommendations.append(f"Directional accuracy ({directional_accuracy:.2%}) is poor. Model is not useful for predicting direction.")
            elif directional_accuracy < 0.55:
                recommendations.append(f"Directional accuracy ({directional_accuracy:.2%}) is low. Use cautiously for trading signals.")
        elif eval_metrics and 'error' not in eval_metrics :
            recommendations.append("Directional accuracy could not be determined. This metric is crucial.")

        # Access quality_score from a fresh call to _assess_model_quality to avoid recursion issues
        # This requires passing significance_check and eval_metrics again.
        # To prevent direct recursion, this part is simplified or assumes quality_score is passed if needed elsewhere.
        # For this function's scope, we can infer based on the conditions already checked.
        if not is_random_walk_equivalent and (
            (significance_check.get('significance_ratio', 1.0) < 0.5 and significance_check.get('total_params',0) > 0) or
            (pd.notna(directional_accuracy) and directional_accuracy < 0.55)
           ):
            recommendations.append("Overall model quality may not be high. Explore alternative orders or techniques.")
            recommendations.append("Ensure data is properly preprocessed (stationarity, outliers).")
        
        if not recommendations:
            recommendations.append("Model diagnostics appear reasonable. Validate with out-of-sample performance.")
            
        return list(set(recommendations))


class ARIMAPortfolioForecaster:
    """
    ARIMA forecasting for multiple stocks in a portfolio.
    COMPLETED: Ensured full class implementation.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 significance_level: float = 0.05):
        """
        Initialize ARIMAPortfolioForecaster.
        Args:
            max_p (int): Max AR order for individual stock models.
            max_d (int): Max differencing order.
            max_q (int): Max MA order.
            significance_level (float): Significance level for parameter validation.
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.significance_level = significance_level
        self.models: Dict[str, ARIMAPredictor] = {} # Stores ARIMAPredictor instances for each symbol
        
    def fit_portfolio(self, stock_data: Dict[str, pd.Series]) -> Dict[str, Dict[str, any]]:
        """
        Fit ARIMA models for multiple stocks.
        
        Args:
            stock_data (Dict[str, pd.Series]): Dictionary of stock price series (symbol -> series)
            
        Returns:
            Dict[str, Dict[str, any]]: Fitting results for each stock, including model summary.
        """
        results = {}
        
        for symbol, price_series in stock_data.items():
            logger.info(f"Fitting ARIMA model for {symbol}")
            
            if price_series.empty or len(price_series) < 20: # Min length for ARIMA
                logger.warning(f"Insufficient data for {symbol} ({len(price_series)} points). Skipping.")
                results[symbol] = {
                    'success': False,
                    'error': 'Insufficient data points for ARIMA fitting.',
                    'model_summary': None,
                    'data_quality': self._assess_data_quality(price_series)
                }
                continue

            try:
                model_predictor = ARIMAPredictor(
                    max_p=self.max_p, 
                    max_d=self.max_d, 
                    max_q=self.max_q, 
                    significance_level=self.significance_level
                )
                
                processed_series = self._preprocess_series(price_series, symbol)
                
                if processed_series is not None and not processed_series.empty:
                    model_predictor.fit(processed_series) # Fit the model
                    
                    if model_predictor.is_fitted:
                        self.models[symbol] = model_predictor # Store the fitted predictor
                        model_summary = model_predictor.get_model_summary()
                        results[symbol] = {
                            'success': True,
                            # 'model_predictor_instance': model_predictor, # Optionally return instance
                            'model_summary': model_summary,
                            'data_quality': self._assess_data_quality(processed_series)
                        }
                        logger.info(f"Successfully fitted ARIMA for {symbol}. Order: {model_predictor.best_order}")
                    else:
                        results[symbol] = {
                            'success': False,
                            'error': 'ARIMA model fitting failed after attempts.',
                            'model_summary': model_predictor.get_model_summary(), # Get summary even if fitting failed (e.g. fallback summary)
                            'data_quality': self._assess_data_quality(processed_series)
                        }
                        logger.warning(f"Failed to fit ARIMA model for {symbol} after all attempts.")
                else:
                    results[symbol] = {
                        'success': False,
                        'error': 'Data preprocessing failed or resulted in empty series.',
                        'model_summary': None,
                        'data_quality': self._assess_data_quality(price_series) # Assess raw if preprocess fails
                    }
                    logger.warning(f"Preprocessing failed for {symbol}. Skipping ARIMA.")
                    
            except Exception as e:
                logger.error(f"Failed to fit ARIMA for {symbol}: {e}", exc_info=True)
                results[symbol] = {
                    'success': False,
                    'error': str(e),
                    'model_summary': None,
                    'data_quality': self._assess_data_quality(price_series)
                }
        
        return results
    
    def _preprocess_series(self, series: pd.Series, symbol: str) -> Optional[pd.Series]:
        """
        Preprocess time series data for better ARIMA fitting.
        """
        try:
            if series.empty:
                logger.warning(f"Input series for {symbol} is empty.")
                return None

            clean_series = series.dropna()
            if clean_series.empty:
                logger.warning(f"Series for {symbol} is empty after dropna.")
                return None
            
            clean_series = clean_series[np.isfinite(clean_series)]
            if clean_series.empty:
                logger.warning(f"Series for {symbol} is empty after removing non-finite values.")
                return None
            
            if len(clean_series) < 20: # Increased min length for robust ARIMA
                logger.warning(f"Insufficient data for {symbol} after cleaning: {len(clean_series)} points. Min 20 required.")
                return None
            
            if clean_series.nunique() < 3: # Need some variation
                logger.warning(f"Series for {symbol} has too few unique values ({clean_series.nunique()}). May not be suitable for ARIMA.")
                # Allow to proceed but with caution, fitting might select (0,0,0) or (0,1,0)
            
            # Outlier handling (simple clipping for robustness)
            q_low = clean_series.quantile(0.01)
            q_high = clean_series.quantile(0.99)
            # Only clip if quantiles are valid numbers
            if pd.notna(q_low) and pd.notna(q_high) and q_low < q_high:
                 clean_series = clean_series.clip(lower=q_low, upper=q_high)
            
            return clean_series
            
        except Exception as e:
            logger.error(f"Error preprocessing series for {symbol}: {e}")
            return None
    
    def _assess_data_quality(self, series: Optional[pd.Series]) -> Dict[str, any]:
        """
        Assess quality of time series data.
        """
        if series is None or series.empty:
            return {'quality_score': 0.0, 'error': 'Empty or None series provided', 'length': 0}

        try:
            quality_metrics: Dict[str, any] = {
                'length': len(series),
                'unique_values': series.nunique(),
                'missing_ratio_original': np.nan, # Cannot determine from already cleaned series
                'volatility': series.pct_change().std() if len(series) > 1 else 0.0,
            }
            if len(series) > 1 and series.iloc[0] != 0 : # Avoid division by zero
                 quality_metrics['trend_strength'] = abs(series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) if series.iloc[0] !=0 else np.nan
            else:
                 quality_metrics['trend_strength'] = np.nan


            score = 100.0
            if quality_metrics['length'] < 50: score -= 30 # Penalize short series heavily
            if quality_metrics['unique_values'] < 5: score -= 25
            if pd.notna(quality_metrics['volatility']) and quality_metrics['volatility'] > 0.1: score -= 10 # Very high daily volatility
            
            quality_metrics['quality_score'] = max(0.0, score)
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Data quality assessment error: {e}")
            return {'quality_score': 30.0, 'error': str(e), 'length': len(series) if series is not None else 0} # Lower score on error
    
    def forecast_portfolio(self, steps: int = 5, confidence_level: float = 0.95) -> Dict[str, Dict[str, any]]:
        """
        Generate forecasts for all fitted models in the portfolio.
        
        Args:
            steps (int): Number of steps to forecast.
            confidence_level (float): Confidence level for prediction intervals.
            
        Returns:
            Dict[str, Dict[str, any]]: Forecast results for each stock (symbol -> forecast_dict).
        """
        forecasts: Dict[str, Dict[str, any]] = {}
        
        if not self.models:
            logger.warning("No models have been fitted for the portfolio. Cannot generate forecasts.")
            return forecasts

        for symbol, model_predictor_instance in self.models.items():
            logger.info(f"Generating forecast for {symbol}...")
            try:
                if model_predictor_instance.is_fitted:
                    forecast_result = model_predictor_instance.forecast(steps=steps, confidence_level=confidence_level)
                    forecasts[symbol] = forecast_result
                else:
                    logger.warning(f"Model for {symbol} is not fitted. Skipping forecast.")
                    forecasts[symbol] = {'error': 'Model not fitted'}
            except Exception as e:
                logger.error(f"Failed to forecast {symbol}: {e}", exc_info=True)
                forecasts[symbol] = {'error': str(e)}
        
        return forecasts

    def evaluate_portfolio_models(self, test_stock_data: Dict[str, pd.Series], forecast_steps: Optional[int] = None) -> Dict[str, Dict[str, any]]:
        """
        Evaluate all fitted ARIMA models in the portfolio using test data.

        Args:
            test_stock_data (Dict[str, pd.Series]): Dictionary of test price series (symbol -> series).
            forecast_steps (Optional[int]): Number of steps to forecast for evaluation. 
                                            If None, uses length of test data for each symbol.

        Returns:
            Dict[str, Dict[str, any]]: Evaluation metrics for each stock.
        """
        eval_results: Dict[str, Dict[str, any]] = {}

        if not self.models:
            logger.warning("No models have been fitted for the portfolio. Cannot evaluate.")
            return eval_results

        for symbol, model_predictor_instance in self.models.items():
            logger.info(f"Evaluating model for {symbol}...")
            if symbol not in test_stock_data or test_stock_data[symbol].empty:
                logger.warning(f"No test data provided for {symbol}. Skipping evaluation.")
                eval_results[symbol] = {'error': 'No test data provided'}
                continue
            
            test_series = self._preprocess_series(test_stock_data[symbol], symbol) # Preprocess test data similarly
            if test_series is None or test_series.empty:
                logger.warning(f"Test data for {symbol} became empty after preprocessing. Skipping evaluation.")
                eval_results[symbol] = {'error': 'Test data empty after preprocessing'}
                continue

            try:
                if model_predictor_instance.is_fitted:
                    # Determine forecast steps for this specific symbol
                    current_forecast_steps = forecast_steps if forecast_steps is not None else len(test_series)
                    evaluation = model_predictor_instance.evaluate(test_series, forecast_steps=current_forecast_steps)
                    eval_results[symbol] = evaluation
                else:
                    logger.warning(f"Model for {symbol} is not fitted. Skipping evaluation.")
                    eval_results[symbol] = {'error': 'Model not fitted'}
            except Exception as e:
                logger.error(f"Failed to evaluate model for {symbol}: {e}", exc_info=True)
                eval_results[symbol] = {'error': str(e)}
        
        return eval_results