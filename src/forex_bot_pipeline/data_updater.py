import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
import time
from src.utils.paths import DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataUpdater:
    """
    Updates database with latest forex data before predictions.
    """

    def __init__(self, db_path, max_retries=5, retry_delay=1):
        self.db_path = db_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # Initial delay in seconds

    def _connect_db(self):
        """Connect to SQLite DB with timeout and enable WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=10.0, detect_types=sqlite3.PARSE_DECLTYPES)
        # Enable WAL mode to improve concurrent access
        conn.execute('PRAGMA journal_mode=WAL;')
        # Optional: Set busy timeout explicitly (in ms)
        conn.execute('PRAGMA busy_timeout=5000;')
        return conn

    def get_latest_date_in_db(self, symbol):
        """Get the most recent date for a symbol in database."""
        try:
            with self._connect_db() as conn:
                query = """
                    SELECT MAX(p.date) 
                    FROM prices p
                    JOIN tickers t ON p.ticker_id = t.ticker_id
                    WHERE t.symbol = ?
                """
                result = pd.read_sql(query, conn, params=(symbol,))
                latest_date = result.iloc[0, 0]

                if latest_date:
                    return pd.to_datetime(latest_date)
            return None

        except sqlite3.Error as e:
            logger.error(f"SQLite error in get_latest_date_in_db for {symbol}: {e}")
            return None

    def fetch_new_data(self, symbol, start_date):
        """Fetch data from start_date to now."""
        end_date = datetime.now()

        ticker = yf.Ticker(symbol)
        fetch_start_date = start_date - timedelta(days=2)
        df = ticker.history(start=fetch_start_date, end=end_date, interval='1h')

        if df.empty:
            return None

        # Format dataframe columns
        df.reset_index(inplace=True)
        df.columns = [col.lower() for col in df.columns]
        df['symbol'] = symbol
        df.rename(columns={'datetime': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] > start_date - timedelta(days=1)]

        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

    def update_symbol(self, symbol):
        """Update a single symbol with latest data."""
        latest_date = self.get_latest_date_in_db(symbol)

        if latest_date is None:
            logger.warning(f"{symbol}: No data in database, skipping update")
            return False

        start_date = latest_date + timedelta(days=1)
        logger.info(f"Updating {symbol} from {start_date.date()} to now...")

        new_data = self.fetch_new_data(symbol, start_date)
        if new_data is None or len(new_data) == 0:
            logger.info(f"{symbol}: Already up to date!")
            return True

        # Rename columns to match database schema
        new_data.rename(columns={
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price'
        }, inplace=True)

        attempt = 0
        while attempt < self.max_retries:
            try:
                with self._connect_db() as conn:
                    query = "SELECT ticker_id FROM tickers WHERE symbol = ?"
                    ticker_id_df = pd.read_sql(query, conn, params=(symbol,))
                    if ticker_id_df.empty:
                        logger.error(f"No ticker_id found for symbol {symbol}. Skipping insert.")
                        return False
                    ticker_id = ticker_id_df.iloc[0, 0]

                    new_data['ticker_id'] = ticker_id
                    new_data['timeframe'] = '1h'
                    new_data.drop(columns=['symbol'], inplace=True)

                    cols_order = ['ticker_id', 'date', 'timeframe', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
                    new_data_ordered = new_data[cols_order]

                    # Insert data in a short transaction
                    conn.execute('BEGIN IMMEDIATE')
                    new_data_ordered.to_sql('prices', conn, if_exists='append', index=False)
                    conn.commit()

                logger.info(f"✓ {symbol}: Added {len(new_data_ordered)} new records")
                return True

            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    wait_time = self.retry_delay * (2 ** attempt)  # exponential backoff
                    logger.warning(f"Database is locked, retrying {symbol} in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    logger.error(f"SQLite OperationalError for {symbol}: {e}")
                    return False

            except Exception as e:
                logger.error(f"General error during update for {symbol}: {e}")
                return False

        logger.error(f"Max retries reached for {symbol}, update failed due to locked database.")
        return False

    def update_all_symbols(self, symbols):
        logger.info(f"{'=' * 60}")
        logger.info("UPDATING DATABASE WITH LATEST DATA")
        logger.info(f"{'=' * 60}")

        for symbol in symbols:
            try:
                self.update_symbol(symbol)
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")

        logger.info("✓ Database update complete")


if __name__ == "__main__":
    updater = DataUpdater(str(DB_PATH))
    symbols = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
        "EURJPY=X", "GBPJPY=X", "GC=F"
    ]
    updater.update_all_symbols(symbols)