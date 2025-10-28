import pandas as pd
import pandas_ta as ta


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

        # --- 1. TREND AND MOMENTUM (Original Features) ---
        data['SMA_Short'] = data['close'].rolling(window=self.short_ma).mean()
        data['SMA_Long'] = data['close'].rolling(window=self.long_ma).mean()
        data['MA_Spread'] = data['SMA_Short'] - data['SMA_Long']

        data.ta.rsi(close=data['close'], length=self.short_rsi, append=True)
        data.ta.rsi(close=data['close'], length=self.medium_rsi, append=True)
        data['RSI_Short'] = data[f'RSI_{self.short_rsi}']
        data['RSI_Medium'] = data[f'RSI_{self.medium_rsi}']
        data.drop(columns=[f'RSI_{self.short_rsi}', f'RSI_{self.medium_rsi}'], errors='ignore', inplace=True)

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

        # --- 2. VOLATILITY, COMMITMENT, & EXHAUSTION (NEW Features) ---

        # Volatility Base (Use the ADX-calculated ATR for a consistent period)
        data['ATR_Base'] = data[f'ATR_{self.adx_period}']

        # Volatility Ratio: Relative risk/reward gauge
        data['Vol_Ratio'] = data['ATR_Base'] / data['close']

        # Commitment: Size of the candle body
        data['Body_Size'] = abs(data['close'] - data['open'])

        # Exhaustion: CCI (Medium RSI period used as length)
        # Note: pandas_ta CCI requires High, Low, Close (HLC)
        cci_series = ta.cci(high=data['high'], low=data['low'], close=data['close'], length=self.medium_rsi, append=False)
        cc_column_name = cci_series.name
        data['CCI'] = cci_series

        # Acceleration: Rate of Change (e.g., over 5 periods)
        data['Close_ROC_5'] = data['close'].pct_change(5)

        # Time features
        data['Hour'] = data.index.hour
        data['DayOfWeek'] = data.index.dayofweek

        # Drop NaNs due to rolling/technical indicator calculations

        data.fillna(method='ffill',inplace=True)
        data.dropna(inplace=True)

        # Select final features
        feature_columns = [
            'MA_Spread', 'RSI_Short', 'RSI_Medium',
            'ADX', 'ADX_Spread',
            'Vol_Ratio', 'Body_Size', 'CCI', 'Close_ROC_5',
            'Hour', 'DayOfWeek',
            'Target_Y'
        ]


        data = data[feature_columns]

        return data