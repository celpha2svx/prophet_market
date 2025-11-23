import pandas as pd
import numpy as np
from datetime import time
import logging
from pathlib import Path
from src.data_pipeline.forex_cl_data import DataProcessor
from src.utils.paths import DB_PATH,SESSION_DATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionDataBuilder:
    """
    Transforms hourly forex data into session-level OHLC data.
    Sessions: London (8am-12pm GMT), NY (1pm-5pm GMT)
    """

    def __init__(self, data_processor):
        """
        Args:
            data_processor: Your existing DataProcessor instance
        """
        self.data_processor = data_processor

        # Session definitions (GMT times)
        self.london_start = time(8, 0)
        self.london_end = time(12, 0)
        self.ny_start = time(13, 0)
        self.ny_end = time(17, 0)

    def load_all_pairs(self, symbols):
        """Load data for all pairs from database."""
        logger.info(f"Loading data for {len(symbols)} pairs...")

        all_data = []
        for symbol in symbols:
            try:
                # Use your existing data loader
                data = self.data_processor.load_forex_data([symbol])

                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
                    logger.info(f"✓ Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"✗ No data for {symbol}")

            except Exception as e:
                logger.error(f"✗ Error loading {symbol}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=False)
            logger.info(f"Total records loaded: {len(combined)}")
            return combined
        else:
            raise ValueError("No data loaded for any symbols!")

    def assign_session(self, df):
        """
        Assigns session labels (London/NY/Other) to each hourly bar.
        """
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # Extract hour from index
        df['hour'] = df.index.hour

        # Assign session based on hour (GMT)
        def get_session(hour):
            if 8 <= hour < 12:
                return 'London'
            elif 13 <= hour < 17:
                return 'NY'
            else:
                return 'Other'

        df['session'] = df['hour'].apply(get_session)

        return df

    def create_session_ohlc(self, df):
        """
        Groups hourly data into session-level OHLC.
        Each session gets: Open, High, Low, Close
        """
        df = df.copy()

        # Filter only London and NY sessions
        session_data = df[df['session'].isin(['London', 'NY'])].copy()

        if session_data.empty:
            logger.warning("No London/NY session data found!")
            return pd.DataFrame()

        # Create session date (the date of the session)
        session_data['session_date'] = session_data.index.date

        # Group by symbol, session_date, and session
        grouped = session_data.groupby(['symbol', 'session_date', 'session'])

        # Calculate OHLC for each session
        session_ohlc = grouped.agg({
            'open': 'first',  # First open of session
            'high': 'max',  # Highest high
            'low': 'min',  # Lowest low
            'close': 'last',  # Last close of session
            'volume': 'sum'  # Total volume (if available)
        }).reset_index()

        # Rename for clarity
        session_ohlc.rename(columns={
            'open': 'session_open',
            'high': 'session_high',
            'low': 'session_low',
            'close': 'session_close',
            'volume': 'session_volume'
        }, inplace=True)

        # Calculate session metrics
        session_ohlc['session_range'] = session_ohlc['session_high'] - session_ohlc['session_low']
        session_ohlc['session_body'] = session_ohlc['session_close'] - session_ohlc['session_open']
        session_ohlc['session_body_pct'] = (session_ohlc['session_body'] / session_ohlc['session_open']) * 100

        # Create target: Bullish (1) if close > open, Bearish (0) otherwise
        session_ohlc['target'] = (session_ohlc['session_close'] > session_ohlc['session_open']).astype(int)

        # Sort by date and session
        session_ohlc['session_date'] = pd.to_datetime(session_ohlc['session_date'])
        session_ohlc.sort_values(['symbol', 'session_date', 'session'], inplace=True)

        logger.info(f"Created {len(session_ohlc)} session records")

        return session_ohlc

    def add_session_sequence_id(self, df):
        """
        Adds a unique sequential ID for each session (useful for tracking).
        """
        df = df.copy()

        # Create datetime for sorting
        df['session_hour'] = df['session'].map({'London': 8, 'NY': 13})
        df['session_datetime'] = pd.to_datetime(df['session_date']) + pd.to_timedelta(df['session_hour'], unit='h')

        # Sort by symbol and datetime
        df.sort_values(['symbol', 'session_datetime'], inplace=True)

        # Add sequence ID per symbol
        df['session_id'] = df.groupby('symbol').cumcount()

        df.drop(columns=['session_hour'], inplace=True)

        return df

    def build_session_dataset(self, symbols, output_path=None):
        """
        Complete pipeline: Load data → Assign sessions → Create OHLC → Save

        Args:
            symbols: List of forex pairs
            output_path: Optional path to save CSV

        Returns:
            DataFrame with session-level data
        """
        logger.info("=" * 60)
        logger.info("BUILDING SESSION DATASET")
        logger.info("=" * 60)

        # Step 1: Load all pair data
        hourly_data = self.load_all_pairs(symbols)

        # Step 2: Assign sessions
        logger.info("\nAssigning sessions to hourly data...")
        hourly_with_sessions = self.assign_session(hourly_data)

        # Step 3: Create session OHLC
        logger.info("\nCreating session-level OHLC...")
        session_data = self.create_session_ohlc(hourly_with_sessions)

        # Step 4: Add sequence IDs
        logger.info("\nAdding session sequence IDs...")
        session_data = self.add_session_sequence_id(session_data)

        # Step 5: Validation
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total sessions: {len(session_data)}")
        logger.info(f"\nSessions per pair:")
        print(session_data.groupby('symbol').size())

        logger.info(f"\nSessions per type:")
        print(session_data['session'].value_counts())

        logger.info(f"\nTarget distribution:")
        print(session_data['target'].value_counts(normalize=True))

        logger.info(f"\nDate range:")
        logger.info(f"  Start: {session_data['session_date'].min()}")
        logger.info(f"  End: {session_data['session_date'].max()}")

        # Step 6: Save if path provided
        if output_path:
            session_data.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved session data to: {output_path}")

        return session_data


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":

    # Initialize your existing data processor
    db_path = str(DB_PATH)
    data_processor = DataProcessor(db_path=db_path)

    # Define forex pairs
    symbols = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
        "EURJPY=X", "GBPJPY=X", "GC=F"
    ]

    # Build session dataset
    builder = SessionDataBuilder(data_processor)
    session_df = builder.build_session_dataset(
        symbols=symbols,
        output_path=SESSION_DATA  # Optional: save to file
    )

    print("\n✓ Session dataset ready!")
    print(f"Shape: {session_df.shape}")
    print(f"\nFirst few rows:")
    print(session_df.head(10))