import sqlite3
import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from src.utils.paths import  DB_PATH

# Configure logging to show where messages come from
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, db_path: str, stop_loss_multiplier=1.0, take_profit_multiplier=2.0, window=8):
        """
        Initializes the DataProcessor with database path and risk parameters.

        :param db_path: Path to the SQLite database.
        :param stop_loss_multiplier: Multiplier for ATR to set the stop loss (for non-GC=F assets).
        :param take_profit_multiplier: Multiplier for ATR to set the take profit (for non-GC=F assets).
        :param window: Rolling window size for future high/low calculation.
        """
        self.db_path = db_path
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.window = window

    def load_forex_data(self, symbols: list[str]) -> pd.DataFrame:
        """
        Loads price data for a list of symbols from the SQLite database.

        :param symbols: A list of ticker symbols (e.g., ["EURUSD=X", "GC=F"]).
        :return: A DataFrame containing the merged price data.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)

            # Build the placeholders for the symbols list
            placeholders = ','.join('?' for _ in symbols)

            query = f"""
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
                    WHERE t.symbol IN ({placeholders})
                    ORDER BY p.date ASC
                """

            df = pd.read_sql_query(query, conn, params=tuple(symbols))

            if df.empty:
                logger.warning(f"No data found for symbols: {symbols}")
                return df

            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            # Convert to datetime and ensure consistency
            df['date'] = pd.to_datetime(df['date'], utc=True)

            logger.info(f"Loaded {len(df)} records for {symbols}")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbols}: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        # pandas_ta automatically aligns the result to the input DataFrame's index
        return ta.atr(df['high'], df['low'], df['close'], timeperiod=period)

    def create_target(self, symbols):
        """
        Improved target creation: checks which level (TP or SL) is hit FIRST.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []
        for symbol in symbols:
            data = self.load_forex_data([symbol])
            if data.empty:
                continue

            current_window = self.window

            # --- Define Risk Levels ---
            if symbol == "GC=F":
                current_window = 8
                atr_sl = 0.35
                atr_tp = 1.00
            else:
                atr_period = 14
                data['ATR'] = self.calculate_atr(data, period=atr_period)
                atr_sl = self.stop_loss_multiplier * data['ATR']
                atr_tp = self.take_profit_multiplier * data['ATR']

            # Buy Side Levels
            data['BUY_TP'] = data['close'] + atr_tp
            data['BUY_SL'] = data['close'] - atr_sl

            # Sell Side Levels
            data['SELL_TP'] = data['close'] - atr_tp
            data['SELL_SL'] = data['close'] + atr_sl

            # --- NEW APPROACH: Find FIRST hit within window ---
            target_list = []

            for i in range(len(data) - current_window):
                future_highs = data['high'].iloc[i + 1:i + 1 + current_window].values
                future_lows = data['low'].iloc[i + 1:i + 1 + current_window].values

                buy_tp = data['BUY_TP'].iloc[i]
                buy_sl = data['BUY_SL'].iloc[i]
                sell_tp = data['SELL_TP'].iloc[i]
                sell_sl = data['SELL_SL'].iloc[i]

                # Check each future bar in sequence
                buy_signal = False
                sell_signal = False

                for high, low in zip(future_highs, future_lows):
                    # Check BUY outcome
                    if high >= buy_tp:
                        buy_signal = True
                        break
                    if low <= buy_sl:
                        break  # SL hit first, no buy signal

                # Reset and check SELL outcome
                for high, low in zip(future_highs, future_lows):
                    # Check SELL outcome
                    if low <= sell_tp:
                        sell_signal = True
                        break
                    if high >= sell_sl:
                        break  # SL hit first, no sell signal

                # Determine target
                if buy_signal and not sell_signal:
                    target_list.append(1)  # UP
                elif sell_signal and not buy_signal:
                    target_list.append(2)  # DOWN
                elif buy_signal and sell_signal:
                    # Both would profit - check which TP is closer
                    buy_distance = abs(buy_tp - data['close'].iloc[i])
                    sell_distance = abs(sell_tp - data['close'].iloc[i])
                    target_list.append(1 if buy_distance < sell_distance else 2)
                else:
                    target_list.append(0)  # NEUTRAL

            # Pad remaining rows with NaN
            target_list.extend([np.nan] * current_window)
            data['Target_Y'] = target_list

            # Drop NaN targets
            data.dropna(subset=['Target_Y'], inplace=True)

            all_data.append(data)

        if all_data:
            logger.info(f"Successfully processed {len(all_data)} symbols.")
            return pd.concat(all_data, ignore_index=False)
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    db_path = str(DB_PATH)
    pipeline = DataProcessor(db_path=db_path)

    print("\n" + "=" * 50)
    print("Calculating features for multiple stocks...")
    features = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
                "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "GC=F"]

    all_data = pipeline.create_target(features)
    print(all_data['Target_Y'].value_counts(normalize=True))
