"""
Technical Indicators Module for Stock Trading Analysis.
Implements common technical indicators used in trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Collection of technical indicators for stock market analysis.
    """
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices (pd.Series): Price series
            window (int): Moving average window
            
        Returns:
            pd.Series: SMA values
        """
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices (pd.Series): Price series
            window (int): EMA window
            
        Returns:
            pd.Series: EMA values
        """
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, 
                       num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pd.Series): Price series
            window (int): Moving average window
            num_std (float): Number of standard deviations
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band (SMA), lower band
        """
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            window (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices (pd.Series): Price series
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            k_period (int): %K period
            d_period (int): %D period
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K values, %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): Lookback period
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): CCI period
            
        Returns:
            pd.Series: CCI values
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate Price Momentum.
        
        Args:
            prices (pd.Series): Price series
            window (int): Momentum period
            
        Returns:
            pd.Series: Momentum values
        """
        return prices - prices.shift(window)
    
    @staticmethod
    def rate_of_change(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        Args:
            prices (pd.Series): Price series
            window (int): ROC period
            
        Returns:
            pd.Series: ROC values
        """
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close (pd.Series): Close prices
            volume (pd.Series): Volume data
            
        Returns:
            pd.Series: OBV values
        """
        price_change = close.diff()
        
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def accumulation_distribution_line(high: pd.Series, low: pd.Series, 
                                      close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            volume (pd.Series): Volume data
            
        Returns:
            pd.Series: A/D Line values
        """
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            volume (pd.Series): Volume data
            window (int): MFI period
            
        Returns:
            pd.Series: MFI values
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        price_change = typical_price.diff()
        
        positive_flow = raw_money_flow.where(price_change > 0, 0)
        negative_flow = raw_money_flow.where(price_change < 0, 0)
        
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                     acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            acceleration (float): Acceleration factor
            maximum (float): Maximum acceleration factor
            
        Returns:
            pd.Series: Parabolic SAR values
        """
        sar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        af = pd.Series(index=close.index, dtype=float)
        ep = pd.Series(index=close.index, dtype=float)
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        af.iloc[0] = acceleration
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(close)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if low.iloc[i] <= sar.iloc[i]:
                    # Trend reversal
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if high.iloc[i] >= sar.iloc[i]:
                    # Trend reversal
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar


class TechnicalIndicatorSuite:
    """
    Comprehensive suite for calculating multiple technical indicators.
    """
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available technical indicators for a stock dataset.
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with all technical indicators
        """
        logger.info("Calculating comprehensive technical indicators")
        
        df_indicators = df.copy()
        
        # Required columns
        required_ohlc = ['open', 'high', 'low', 'close']
        price_col = 'close'
        if 'adjusted_close' in df.columns:
            price_col = 'adjusted_close'
        
        prices = df_indicators[price_col]
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df_indicators[f'sma_{window}'] = TechnicalIndicators.simple_moving_average(prices, window)
            df_indicators[f'ema_{window}'] = TechnicalIndicators.exponential_moving_average(prices, window)
        
        # Bollinger Bands
        for window in [10, 20]:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, window)
            df_indicators[f'bb_upper_{window}'] = upper
            df_indicators[f'bb_middle_{window}'] = middle
            df_indicators[f'bb_lower_{window}'] = lower
            df_indicators[f'bb_width_{window}'] = (upper - lower) / middle
            df_indicators[f'bb_position_{window}'] = (prices - lower) / (upper - lower)
        
        # RSI
        for window in [14, 21]:
            df_indicators[f'rsi_{window}'] = TechnicalIndicators.rsi(prices, window)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
        df_indicators['macd'] = macd_line
        df_indicators['macd_signal'] = signal_line
        df_indicators['macd_histogram'] = histogram
        
        # Momentum indicators
        for window in [5, 10, 20]:
            df_indicators[f'momentum_{window}'] = TechnicalIndicators.momentum(prices, window)
            df_indicators[f'roc_{window}'] = TechnicalIndicators.rate_of_change(prices, window)
        
        # OHLC-based indicators
        if all(col in df_indicators.columns for col in required_ohlc):
            high = df_indicators['high']
            low = df_indicators['low']
            close = df_indicators['close']
            
            # Stochastic Oscillator
            k_percent, d_percent = TechnicalIndicators.stochastic_oscillator(high, low, close)
            df_indicators['stoch_k'] = k_percent
            df_indicators['stoch_d'] = d_percent
            
            # Williams %R
            df_indicators['williams_r'] = TechnicalIndicators.williams_r(high, low, close)
            
            # ATR
            df_indicators['atr'] = TechnicalIndicators.average_true_range(high, low, close)
            
            # CCI
            df_indicators['cci'] = TechnicalIndicators.commodity_channel_index(high, low, close)
            
            # Parabolic SAR
            df_indicators['parabolic_sar'] = TechnicalIndicators.parabolic_sar(high, low, close)
        
        # Volume-based indicators
        if 'volume' in df_indicators.columns:
            volume = df_indicators['volume']
            
            # OBV
            df_indicators['obv'] = TechnicalIndicators.on_balance_volume(prices, volume)
            
            # Volume moving averages
            for window in [10, 20, 50]:
                df_indicators[f'volume_sma_{window}'] = volume.rolling(window=window).mean()
            
            if all(col in df_indicators.columns for col in required_ohlc):
                high = df_indicators['high']
                low = df_indicators['low']
                
                # A/D Line
                df_indicators['ad_line'] = TechnicalIndicators.accumulation_distribution_line(
                    high, low, prices, volume)
                
                # MFI
                df_indicators['mfi'] = TechnicalIndicators.money_flow_index(
                    high, low, prices, volume)
        
        # Additional derived indicators
        self._add_derived_signals(df_indicators)
        
        logger.info(f"Added {len(df_indicators.columns) - len(df.columns)} technical indicators")
        return df_indicators
    
    def _add_derived_signals(self, df: pd.DataFrame):
        """Add derived trading signals based on technical indicators."""
        
        # Moving average crossover signals
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_crossover_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_crossover_12_26'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # RSI signals
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands signals
        if all(col in df.columns for col in ['bb_upper_20', 'bb_lower_20']):
            price_col = 'adjusted_close' if 'adjusted_close' in df.columns else 'close'
            df['bb_squeeze'] = (df['bb_width_20'] < df['bb_width_20'].rolling(20).mean()).astype(int)
            df['bb_breakout_upper'] = (df[price_col] > df['bb_upper_20']).astype(int)
            df['bb_breakout_lower'] = (df[price_col] < df['bb_lower_20']).astype(int)
        
        # Stochastic signals
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            df['stoch_oversold'] = ((df['stoch_k'] < 20) & (df['stoch_d'] < 20)).astype(int)
            df['stoch_overbought'] = ((df['stoch_k'] > 80) & (df['stoch_d'] > 80)).astype(int)
    
    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary of current technical signals.
        
        Args:
            df (pd.DataFrame): Data with technical indicators
            
        Returns:
            dict: Signal summary
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        # Trend signals
        trend_score = 0
        trend_count = 0
        
        if 'ma_crossover_20_50' in df.columns:
            signals['ma_trend'] = 'bullish' if latest['ma_crossover_20_50'] else 'bearish'
            trend_score += latest['ma_crossover_20_50']
            trend_count += 1
        
        if 'macd_bullish' in df.columns:
            signals['macd_trend'] = 'bullish' if latest['macd_bullish'] else 'bearish'
            trend_score += latest['macd_bullish']
            trend_count += 1
        
        # Momentum signals
        momentum_signals = []
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            if rsi < 30:
                momentum_signals.append('oversold')
            elif rsi > 70:
                momentum_signals.append('overbought')
            else:
                momentum_signals.append('neutral')
        
        # Overall assessment
        if trend_count > 0:
            signals['overall_trend'] = 'bullish' if trend_score / trend_count > 0.5 else 'bearish'
        
        signals['momentum'] = momentum_signals
        
        return signals