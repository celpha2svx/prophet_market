import sqlite3
import pandas as pd
import pandas_ta as ta
import logging
import numpy as np

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
        Loads data, calculates risk levels (SL/TP), and defines the Target_Y column (0: Hold, 1: Buy, 2: Sell).

        :param symbols: A list or single string of ticker symbols to process.
        :return: A concatenated DataFrame of all processed symbols.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []
        for symbol in symbols:
            # FIX: The key fix - always pass the symbol as a list to load_forex_data
            data = self.load_forex_data([symbol])

            if data.empty:
                continue

            # Use the instance's default window initially
            current_window = self.window

            # --- Define Risk Levels (Stop Loss/Take Profit) ---
            if symbol == "GC=F":
                current_window = 8  # Custom window for Gold
                stoploss_pips = 0.35
                takeprofit_pips = 1.00

                # Buy Side Levels
                data['Stop_Loss_Level'] = data['close'] - stoploss_pips
                data['Take_Profit_Level'] = data['close'] + takeprofit_pips

                # Sell Side Levels (Needed for Target_Y=2)
                data['SELL_Target_Low'] = data['close'] - takeprofit_pips
                data['SELL_Stop_Loss_High'] = data['close'] + stoploss_pips
            else:
                atr_period = 14
                data['ATR'] = self.calculate_atr(data, period=atr_period)

                # Buy Side Levels (using ATR multiples)
                data['Stop_Loss_Level'] = data['close'] - self.stop_loss_multiplier * data['ATR']
                data['Take_Profit_Level'] = data['close'] + self.take_profit_multiplier * data['ATR']

                # FIX: Define Sell Side Levels for all other symbols (using ATR multiples)
                # Sell Target is reached when price drops by TP multiplier ATR
                data['SELL_Target_Low'] = data['close'] - self.take_profit_multiplier * data['ATR']
                # Sell Stop Loss is hit when price rises by SL multiplier ATR
                data['SELL_Stop_Loss_High'] = data['close'] + self.stop_loss_multiplier * data['ATR']

            # --- Calculate Future High/Low ---
            # Use the local 'current_window' variable
            data['Future_High_Max'] = data['high'].rolling(window=current_window).max().shift(-current_window)
            data['Future_Low_Min'] = data['low'].rolling(window=current_window).min().shift(-current_window)

            # --- Define Target Conditions ---

            # Condition 1: Buy (Target_Y = 1) - TP hit before SL hit
            up_condition = (data['Future_High_Max'] >= data['Take_Profit_Level']) & \
                           (data['Future_Low_Min'] > data['Stop_Loss_Level'])

            # Condition 2: Sell (Target_Y = 2) - TP hit before SL hit
            down_condition = (data['Future_Low_Min'] <= data['SELL_Target_Low']) & \
                             (data['Future_High_Max'] < data['SELL_Stop_Loss_High'])

            conditions = [up_condition, down_condition]
            choices = [1, 2]
            data['Target_Y'] = np.select(conditions, choices, default=0)

            data.dropna(subset=['Future_High_Max'], inplace=True)

            all_data.append(data)

        # Combine all processed symbol data
        if all_data:
            logger.info(f"Successfully processed and concatenated data for {len(all_data)} symbols.")
            return pd.concat(all_data, ignore_index=True)
        else:
            logger.warning("No data was successfully loaded or processed.")
            return pd.DataFrame()