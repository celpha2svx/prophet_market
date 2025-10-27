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

        data.set_index('date', inplace=True)

        # Moving Averages
        data['SMA_Short'] = data['close'].rolling(window=self.short_ma).mean()
        data['SMA_Long'] = data['close'].rolling(window=self.long_ma).mean()
        data['MA_Spread'] = data['SMA_Short'] - data['SMA_Long']

        # RSI
        data.ta.rsi(close=data['close'], length=self.short_rsi, append=True)
        data.ta.rsi(close=data['close'], length=self.medium_rsi, append=True)

        data['RSI_Short'] = data[f'RSI_{self.short_rsi}']
        data['RSI_Medium'] = data[f'RSI_{self.medium_rsi}']

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

        # Assign feature names
        data['ADX'] = data[f'ADX_{self.adx_period}']
        data['PlusDI'] = data[f'DMP_{self.adx_period}']
        data['MinusDI'] = data[f'DMN_{self.adx_period}']
        data['ADX_Spread'] = data['PlusDI'] - data['MinusDI']
        data['ATR_Feature'] = data[f'ATR_{self.adx_period}']

        # Time features (Now that 'date' is the index)
        data['Hour'] = data.index.hour
        data['DayOfWeek'] = data.index.dayofweek

        # Drop NaNs due to rolling calculations
        data.dropna(inplace=True)


        feature_columns = [
            'MA_Spread', 'RSI_Short', 'RSI_Medium',
            'ADX', 'ADX_Spread', 'Hour', 'DayOfWeek',
            'Target_Y'
        ]

        # Keep only the features you want for the model
        data = data[feature_columns]

        return data