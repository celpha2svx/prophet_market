import sqlite3
import requests
import pandas as pd
from datetime import datetime
import time
from typing import List, Optional
import logging
import os
import yaml
import yfinance as yf
from src.utils.paths import DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_app_config():
    """Load configuration from YAML file and environment variables"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config = os.path.join(script_dir, os.pardir, os.pardir, "config.yaml")
    with open(absolute_config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class DatabaseManager:
    """Handles all database operations for the stock/forex predictor"""

    def __init__(self, db_path: str):
        if not db_path:
            raise ValueError("Db path must be provided to DatabaseManager")
        self.db_path = db_path
        # Close any existing connections first
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.close()
        except:
            pass
        self.init_database()

    def init_database(self):
        """Create all tables and indexes"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT,
                    market TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    sector TEXT,
                    country TEXT,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatedat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker_id INTEGER NOT NULL,
                    date TIMESTAMP NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume BIGINT,
                    adjusted_close REAL,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id),
                    UNIQUE(ticker_id, date, timeframe)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker_id INTEGER NOT NULL,
                    action_date TIMESTAMP NOT NULL,
                    action_type TEXT NOT NULL,
                    value REAL,
                    description TEXT,
                    ex_date TIMESTAMP,
                    record_date TIMESTAMP,
                    payment_date TIMESTAMP,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id)
                )
            """)

            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker_id, date)",
                "CREATE INDEX IF NOT EXISTS idx_prices_timeframe ON prices(timeframe)",
                "CREATE INDEX IF NOT EXISTS idx_tickers_symbol ON tickers(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_corp_actions_ticker_date ON corporate_actions(ticker_id, action_date)"
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            conn.commit()
            logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def add_ticker(self, symbol: str, name: str, market: str, currency: str,
                   sector: Optional[str] = None, country: Optional[str] = None) -> Optional[int]:
        """Add a new ticker and return its ID"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tickers (symbol, name, market, currency, sector, country)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, name, market, currency, sector, country))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Database error in add_ticker: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()

    def get_ticker_id(self, symbol: str) -> Optional[int]:
        """Get ticker ID by symbol"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            cursor.execute("SELECT ticker_id FROM tickers WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Database error in get_ticker_id: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def add_price_data(self, ticker_id: int, date: datetime, timeframe: str,
                       open_price: float, high_price: float, low_price: float,
                       close_price: float, volume: Optional[int] = None, adjusted_close: Optional[float] = None):
        """Add price data for a ticker without unintended replacement"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 1 FROM prices WHERE ticker_id = ? AND date = ? AND timeframe = ?
            """, (ticker_id, date, timeframe))
            exists = cursor.fetchone()

            if not exists:
                cursor.execute("""
                    INSERT INTO prices 
                    (ticker_id, date, timeframe, open_price, high_price, low_price, close_price, volume, adjusted_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker_id, date, timeframe, open_price, high_price, low_price, close_price, volume,
                      adjusted_close))
                conn.commit()
            else:
                logger.debug(f"Price data for ticker {ticker_id} on {date} already exists. Skipping insert.")

        except sqlite3.Error as e:
            logger.error(f"Database error in add_price_data: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()


class DataScraper:
    """Handles data scraping from Yahoo Finance"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_yahoo_finance_data(self, symbol: str, period: str = "1y", max_retries: int = 3):
        """Scrape data from Yahoo Finance using yfinance library"""
        attempt = 0
        while attempt < max_retries:
            try:
                ticker = yf.Ticker(symbol)
                if symbol.endswith("=X") or symbol == "GC=F":
                    interval = "60m"
                else:
                    interval = "1d"

                data = ticker.history(period=period, interval=interval)

                if data.empty:
                    logger.error(f"No data found for {symbol} (attempt {attempt + 1}/{max_retries})")
                    attempt += 1
                    time.sleep(2 ** attempt)
                    continue

                ticker_id = self.db_manager.get_ticker_id(symbol)
                if not ticker_id:
                    info = ticker.info
                    market = "FOREX" if symbol.endswith("=X") else info.get('exchange', 'UNKNOWN')
                    ticker_id = self.db_manager.add_ticker(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        market=market,
                        currency=info.get('currency', 'USD'),
                        sector=info.get('sector'),
                        country=info.get('country')
                    )

                records_added = 0
                for date, row in data.iterrows():
                    try:
                        if hasattr(date, 'to_pydatetime'):
                            date_obj = date.to_pydatetime()
                        else:
                            date_obj = datetime.strptime(str(date).split()[0], '%Y-%m-%d')

                        self.db_manager.add_price_data(
                            ticker_id=ticker_id,
                            date=date_obj,
                            timeframe='daily',
                            open_price=float(row['Open']),
                            high_price=float(row['High']),
                            low_price=float(row['Low']),
                            close_price=float(row['Close']),
                            volume=int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else None,
                            adjusted_close=float(row['Close'])
                        )
                        records_added += 1
                    except Exception as e:
                        logger.error(f"Error processing date {date}: {str(e)}")
                        continue

                logger.info(f"Successfully scraped {records_added} records for {symbol}")
                return True

            except Exception as e:
                logger.error(f"Error scraping Yahoo Finance data for {symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                attempt += 1
                time.sleep(2 ** attempt)

        logger.error(f"Failed to scrape data for {symbol} after {max_retries} attempts")
        return False


class DataIngestionManager:
    """Main class to orchestrate data ingestion"""

    def __init__(self, db_path: str):
        self.db_manager = DatabaseManager(db_path=db_path)
        self.scraper = DataScraper(self.db_manager)

    def ingest_stock_data(self, symbols: List[str], data_source: str = "yahoo", **kwargs):
        """Ingest data for multiple stock symbols"""
        results = {}

        for symbol in symbols:
            logger.info(f"Starting data ingestion for {symbol}")

            if data_source == "yahoo":
                period = kwargs.get('period', '1y')
                success = self.scraper.scrape_yahoo_finance_data(symbol, period=period)
            else:
                logger.error(f"Unsupported data source: {data_source}")
                success = False

            results[symbol] = success

            time.sleep(1)

        return results

    def get_sample_data(self):
        """Add some sample tickers for testing"""
        sample_tickers = [
            ("AAPL", "Apple Inc.", "NASDAQ", "USD", "Technology", "USA"),
            ("MSFT", "Microsoft Corporation", "NASDAQ", "USD", "Technology", "USA"),
            ("EURUSD=X", "Euro/USD", "FOREX", "USD", None, None)
        ]

        for symbol, name, market, currency, sector, country in sample_tickers:
            self.db_manager.add_ticker(symbol, name, market, currency, sector, country)

        logger.info("Sample tickers added to database")


def main():
    try:
        cfg = load_app_config()
        db_path = str(DB_PATH)
        ingestion_manager = DataIngestionManager(db_path=db_path)

        ingestion_manager.get_sample_data()

        source = cfg["app"]["data_source"]
        stock_symbols = cfg["tickers"]["stocks"] + cfg["tickers"]["forex"]
        yahoo_period = cfg["ingestion"]["yahoo"]["period"]

        print(f"Testing data source {source} ingestion...")

        if source == "yahoo":
            results = ingestion_manager.ingest_stock_data(
                stock_symbols,
                data_source="yahoo",
                period=yahoo_period
            )
        else:
            logger.error(f"Unsupported data source in config: {source}")
            results = {}

        print("Ingestion results:", results)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()