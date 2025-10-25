import sqlite3
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from src.utils.paths import ALL_FEATURE_CSV, DB_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class StockFeaturePipeline:
    """
    Feature engineering pipeline for stock data.
    Calculates technical indicators and risk metrics.
    """

    def __init__(self, db_path: str, version: str = "v1.0"):
        self.db_path = db_path
        self.version = version
        # Configuration for indicators
        self.sma_windows = [5, 20, 50]
        self.ema_windows = [12, 26]
        self.rsi_window = 14
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.atr_window = 14
        self.volatility_window = 20
        self.BB = 20
        self.Vol = 20

    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load price data for a specific stock symbol from database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT 
                    p.date,
                    p.open_price,
                    p.high_price,
                    p.low_price,
                    p.close_price,
                    p.volume,
                    t.symbol
                FROM prices p
                JOIN tickers t ON p.ticker_id = t.ticker_id
                WHERE t.symbol = ?
                ORDER BY p.date ASC
            """

            df = pd.read_sql_query(query, conn, params=(symbol,))

            if df.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return df

            # Rename columns to standard names
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Loaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def load_multiple_stocks(self, symbols: List[str]) -> pd.DataFrame:
        """Load data for multiple stocks"""
        all_data = []
        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if not df.empty:
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    # ============ TECHNICAL INDICATORS ============

    def calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()

    def calculate_ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_BB(self,series:pd.Series, window: int = 20 , num_std: int = 2) -> pd.DataFrame:
        middle_band = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)

        return pd.DataFrame({
            "BB_Middle":middle_band,
            "BB_Upper":upper_band,
            "BB_Lower":lower_band
             })


    def calculate_macd(self, series: pd.Series) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        fast = self.macd_params['fast']
        slow = self.macd_params['slow']
        signal = self.macd_params['signal']

        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })

    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range (Volatility measure)"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    # ============ RETURNS AND RISK METRICS ============

    def calculate_returns(self, series: pd.Series) -> pd.Series:
        """Calculate daily returns (percentage change)"""
        return series.pct_change()

    def calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Rolling volatility (annualized)"""
        return returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_downside_deviation(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Downside deviation (only negative returns)"""
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        return negative_returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_max_drawdown(self, series: pd.Series, window: int = 252) -> pd.Series:
        """Rolling maximum drawdown"""
        rolling_max = series.rolling(window=window, min_periods=1).max()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown

    def calculate_normalized_volume(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate normalized (z-score) volume over a rolling window.
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        normalized_vol = (series - rolling_mean) / rolling_std
        return normalized_vol

    # ============ MAIN TRANSFORM METHOD ============

    def transform(self, symbol: str) -> pd.DataFrame:
        """
        Calculate all features for a given stock symbol.
        Returns a DataFrame with all technical indicators and risk metrics.
        """
        # Load data
        df = self.load_stock_data(symbol)

        if df.empty:
            logger.error(f"Cannot transform empty data for {symbol}")
            return pd.DataFrame()

        # Create features dataframe
        features = df[['date', 'symbol', 'close']].copy()

        # Moving Averages
        for window in self.sma_windows:
            features[f'sma_{window}'] = self.calculate_sma(df['close'], window)

        for window in self.ema_windows:
            features[f'ema_{window}'] = self.calculate_ema(df['close'], window)

        # RSI
        features['rsi'] = self.calculate_rsi(df['close'], self.rsi_window)

        # Bollinger Bands
        bb_df = self.calculate_BB(df['close'], self.BB)
        features["BB_Middle"] = bb_df["BB_Middle"]
        features["BB_Upper"] = bb_df["BB_Upper"]
        features["BB_Lower"] = bb_df["BB_Lower"]



        # MACD
        macd_df = self.calculate_macd(df['close'])
        features['macd'] = macd_df['macd']
        features['macd_signal'] = macd_df['macd_signal']
        features['macd_histogram'] = macd_df['macd_histogram']

        # ATR
        features['atr'] = self.calculate_atr(df, self.atr_window)

        # Returns
        features['daily_return'] = self.calculate_returns(df['close'])

        # Volatility
        features['realized_volatility'] = self.calculate_realized_volatility(
            features['daily_return'], self.volatility_window
        )

        # Risk Metrics
        features['downside_deviation'] = self.calculate_downside_deviation(
            features['daily_return'], self.volatility_window
        )
        features['max_drawdown'] = self.calculate_max_drawdown(df['close'])

        features['volume'] = self.calculate_normalized_volume(df['volume'],self.Vol)

        logger.info(f"Generated {len(features.columns)} features for {symbol}")
        return features

    def transform_multiple(self, symbols: List[str]) -> pd.DataFrame:
        """Transform multiple stocks and combine into one DataFrame"""
        all_features = []
        for symbol in symbols:
            features = self.transform(symbol)
            if not features.empty:
                all_features.append(features)

        if all_features:
            return pd.concat(all_features, ignore_index=True)
        return pd.DataFrame()


# ============ EXAMPLE USAGE ============

def main():
    # Initialize pipeline
    db_path = str(DB_PATH)
    pipeline = StockFeaturePipeline(db_path=db_path)

    # Calculate for multiple stocks
    print("\n" + "=" * 50)
    print("Calculating features for multiple stocks...")
    features =[ "AAPL", "MSFT", "TSLA", "JPM","NVDA","GOOGL", "AMD" ,
                "SNOW" , "COIN","META" ,"BAC" , "GS" ,"MA" , "UNH" ,
                "ABBV" , "CVX" , "WMT" ,"NKE" ,"PFE" , "PG","DIS","SCHW",
                "QCOM", "ADBE","CRM", "BMY","C","SBUX","UPS","BA","MCD",
                "BLK","MDT","TMO", "AVGO","INTC", "AXP","LLY","COST", "CAT","DE" ]


    all_features = pipeline.transform_multiple(features)
    all_features.to_csv(ALL_FEATURE_CSV, index=False)
    print(f"\nTotal records: {len(all_features)}")
    print(f"Symbols: {all_features['symbol'].unique()}")

if __name__ == "__main__":
    main()