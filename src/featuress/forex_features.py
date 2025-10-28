import pandas as pd
import pandas_ta as ta
import numpy as np


class FeatureEngineer:
    def __init__(self, symbol: str):
        if symbol == 'GC=F':
            self.short_ma = 30
            self.long_ma = 150
            self.short_rsi = 10
            self.medium_rsi = 20
            self.adx_period = 20
        else:
            self.short_ma = 20
            self.long_ma = 100
            self.short_rsi = 6
            self.medium_rsi = 14
            self.adx_period = 14

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:

        # --- PREPARATION ---
        data.set_index('date', inplace=True)

        # --- 1. TREND AND MOMENTUM (Enhanced) ---
        data['SMA_Short'] = data['close'].rolling(window=self.short_ma).mean()
        data['SMA_Long'] = data['close'].rolling(window=self.long_ma).mean()
        data['MA_Spread'] = data['SMA_Short'] - data['SMA_Long']

        # NEW: Price position relative to MAs (normalized)
        data['Price_Above_Short_MA'] = (data['close'] - data['SMA_Short']) / data['SMA_Short']
        data['Price_Above_Long_MA'] = (data['close'] - data['SMA_Long']) / data['SMA_Long']

        # RSI
        data.ta.rsi(close=data['close'], length=self.short_rsi, append=True)
        data.ta.rsi(close=data['close'], length=self.medium_rsi, append=True)
        data['RSI_Short'] = data[f'RSI_{self.short_rsi}']
        data['RSI_Medium'] = data[f'RSI_{self.medium_rsi}']

        # NEW: RSI momentum (is RSI rising or falling?)
        data['RSI_Short_Change'] = data['RSI_Short'].diff(3)
        data['RSI_Medium_Change'] = data['RSI_Medium'].diff(5)

        data.drop(columns=[f'RSI_{self.short_rsi}', f'RSI_{self.medium_rsi}'], errors='ignore', inplace=True)

        # ADX
        adx_df = ta.adx(high=data['high'], low=data['low'], close=data['close'], length=self.adx_period, append=False)
        adx_df.columns = [
            f'ADX_{self.adx_period}',
            f'DMP_{self.adx_period}',
            f'DMN_{self.adx_period}',
            f'ATR_{self.adx_period}'
        ]
        data = data.join(adx_df)

        data['ADX'] = data[f'ADX_{self.adx_period}']
        data['PlusDI'] = data[f'DMP_{self.adx_period}']
        data['MinusDI'] = data[f'DMN_{self.adx_period}']
        data['ADX_Spread'] = data['PlusDI'] - data['MinusDI']

        # NEW: ADX trend (is trend strengthening?)
        data['ADX_Change'] = data['ADX'].diff(3)

        # --- 2. VOLATILITY & MOMENTUM (Enhanced) ---
        data['ATR_Base'] = data[f'ATR_{self.adx_period}']
        data['Vol_Ratio'] = data['ATR_Base'] / data['close']

        # NEW: Volatility trend (is volatility expanding or contracting?)
        data['Vol_Ratio_Change'] = data['Vol_Ratio'].diff(5)

        # NEW: Bollinger Bands (volatility + mean reversion)
        bb = ta.bbands(close=data['close'], length=20, std=2, append=False)

        # Safely extract columns (pandas_ta might name them differently)
        if bb is not None and not bb.empty:
            bb_cols = bb.columns.tolist()
            # The columns are usually named BBL_, BBM_, BBU_*
            data['BB_Lower'] = bb[bb_cols[0]]  # First column is lower band
            data['BB_Middle'] = bb[bb_cols[1]]  # Second is middle
            data['BB_Upper'] = bb[bb_cols[2]]  # Third is upper

            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            data['BB_Position'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        else:
            # Fallback if bbands fails
            data['BB_Lower'] = data['close']
            data['BB_Middle'] = data['close']
            data['BB_Upper'] = data['close']
            data['BB_Width'] = 0
            data['BB_Position'] = 0.5

        # Body Size (keep but make it relative)
        data['Body_Size_Pct'] = abs(data['close'] - data['open']) / data['open']

        # CCI
        cci_series = ta.cci(high=data['high'], low=data['low'], close=data['close'], length=self.medium_rsi,
                            append=False)
        data['CCI'] = cci_series

        # --- 3. MULTI-TIMEFRAME MOMENTUM (CRITICAL!) ---
        # Rate of Change across multiple periods
        data['ROC_3'] = data['close'].pct_change(3)  # Very short term
        data['ROC_5'] = data['close'].pct_change(5)  # Short term
        data['ROC_10'] = data['close'].pct_change(10)  # Medium term
        data['ROC_20'] = data['close'].pct_change(20)  # Longer term

        # NEW: Momentum alignment (are multiple timeframes agreeing?)
        data['Momentum_Alignment'] = (
                                             np.sign(data['ROC_3']) +
                                             np.sign(data['ROC_5']) +
                                             np.sign(data['ROC_10'])
                                     ) / 3  # Ranges from -1 (all bearish) to +1 (all bullish)

        # --- 4. PRICE ACTION PATTERNS ---
        # NEW: Higher highs / Lower lows detection
        data['HH'] = (data['high'] > data['high'].shift(1)) & (data['high'].shift(1) > data['high'].shift(2))
        data['LL'] = (data['low'] < data['low'].shift(1)) & (data['low'].shift(1) < data['low'].shift(2))
        data['HH'] = data['HH'].astype(int)
        data['LL'] = data['LL'].astype(int)

        # NEW: Candle patterns (simple but effective)
        data['Upper_Shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['Lower_Shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        data['Upper_Shadow_Pct'] = data['Upper_Shadow'] / (data['high'] - data['low'] + 1e-10)
        data['Lower_Shadow_Pct'] = data['Lower_Shadow'] / (data['high'] - data['low'] + 1e-10)

        # --- 5. VOLUME (if available) ---
        if 'volume' in data.columns:
            data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['volume'] / (data['Volume_SMA'] + 1)

        # --- 6. TIME FEATURES (Improved) ---
        # Instead of raw hour, use session indicators
        data['Hour'] = data.index.hour
        data['Is_Asian_Session'] = ((data['Hour'] >= 0) & (data['Hour'] < 8)).astype(int)
        data['Is_London_Session'] = ((data['Hour'] >= 8) & (data['Hour'] < 16)).astype(int)
        data['Is_NY_Session'] = ((data['Hour'] >= 13) & (data['Hour'] < 22)).astype(int)
        data['DayOfWeek'] = data.index.dayofweek

        # Drop the raw Hour since we have session indicators
        data.drop(columns=['Hour'], inplace=True)

        # --- CLEANUP ---
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        # --- FINAL FEATURE SELECTION ---
        feature_columns = [
            # Trend
            'MA_Spread', 'Price_Above_Short_MA', 'Price_Above_Long_MA',

            # Momentum
            'RSI_Short', 'RSI_Medium', 'RSI_Short_Change', 'RSI_Medium_Change',

            # Trend Strength
            'ADX', 'ADX_Spread', 'ADX_Change',

            # Volatility
            'Vol_Ratio', 'Vol_Ratio_Change', 'BB_Width', 'BB_Position',

            # Multi-timeframe
            'ROC_3', 'ROC_5', 'ROC_10', 'ROC_20', 'Momentum_Alignment',

            # Price Action
            'Body_Size_Pct', 'HH', 'LL', 'Upper_Shadow_Pct', 'Lower_Shadow_Pct',

            # Other
            'CCI',

            # Time
            'Is_Asian_Session', 'Is_London_Session', 'Is_NY_Session', 'DayOfWeek',

            # Target
            'Target_Y'
        ]

        # Add volume features if they exist
        if 'Volume_Ratio' in data.columns:
            feature_columns.insert(-1, 'Volume_Ratio')

        data = data[feature_columns]

        return data