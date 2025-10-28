import pandas as pd
import numpy as np
import logging
from src.utils.paths import SESSION_DATA,FEATURE_SESSION_DATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionFeatureEngineer:
    """
    Creates predictive features for session-level trading.
    Features capture: previous session patterns, trends, volatility, and timing.
    """

    def __init__(self):
        pass

    def add_lag_features(self, df):
        """
        Add features from previous sessions (shifted data).
        These tell us: What happened in recent sessions?
        """
        df = df.copy()

        # Sort to ensure proper chronological order per symbol
        df = df.sort_values(['symbol', 'session_datetime'])

        # Group by symbol for lag features
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Previous session outcome (most important!)
            df.loc[mask, 'prev_session_target'] = symbol_data['target'].shift(1)
            df.loc[mask, 'prev_session_body_pct'] = symbol_data['session_body_pct'].shift(1)
            df.loc[mask, 'prev_session_range'] = symbol_data['session_range'].shift(1)

            # Previous session type
            df.loc[mask, 'prev_was_london'] = (symbol_data['session'].shift(1) == 'London').astype(int)

            # Last 3 sessions momentum
            df.loc[mask, 'last_3_sessions_bullish'] = symbol_data['target'].shift(1).rolling(3).sum()

            # Last 5 sessions trend strength (how many were bullish?)
            df.loc[mask, 'last_5_sessions_bullish_pct'] = symbol_data['target'].shift(1).rolling(5).mean()

            # Session-to-session momentum (is movement accelerating?)
            df.loc[mask, 'body_pct_change'] = symbol_data['session_body_pct'].diff()
            df.loc[mask, 'range_change'] = symbol_data['session_range'].diff()

        return df

    def add_asian_session_features(self, df):
        """
        Add features from Asian session (overnight movement).
        Asian session = 00:00 to 07:00 GMT (before London opens)

        For London predictions: Use Asian session movement
        For NY predictions: Use London session movement (already captured in lag features)
        """
        df = df.copy()

        # For London sessions, we want to know what happened overnight
        # Proxy: difference between current open and previous close
        df = df.sort_values(['symbol', 'session_datetime'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Gap at open (overnight movement)
            df.loc[mask, 'overnight_gap'] = symbol_data['session_open'] - symbol_data['session_close'].shift(1)
            df.loc[mask, 'overnight_gap_pct'] = (df.loc[mask, 'overnight_gap'] / symbol_data['session_close'].shift(
                1)) * 100

        # Only relevant for London sessions
        df['is_london'] = (df['session'] == 'London').astype(int)

        return df

    def add_multi_session_trends(self, df):
        """
        Add trend features over multiple sessions.
        Are we in an uptrend? Downtrend? Consolidation?
        """
        df = df.copy()
        df = df.sort_values(['symbol', 'session_datetime'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Net movement over last N sessions
            df.loc[mask, 'net_move_3_sessions'] = symbol_data['session_body'].shift(1).rolling(3).sum()
            df.loc[mask, 'net_move_5_sessions'] = symbol_data['session_body'].shift(1).rolling(5).sum()
            df.loc[mask, 'net_move_10_sessions'] = symbol_data['session_body'].shift(1).rolling(10).sum()

            # Trend consistency (std of session bodies - lower = more consistent)
            df.loc[mask, 'body_std_5_sessions'] = symbol_data['session_body'].shift(1).rolling(5).std()

            # Average session range (volatility proxy)
            df.loc[mask, 'avg_range_5_sessions'] = symbol_data['session_range'].shift(1).rolling(5).mean()
            df.loc[mask, 'avg_range_10_sessions'] = symbol_data['session_range'].shift(1).rolling(10).mean()

        return df

    def add_volatility_features(self, df):
        """
        Volatility expansion/contraction indicators.
        High volatility = good for trading, Low volatility = avoid
        """
        df = df.copy()
        df = df.sort_values(['symbol', 'session_datetime'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Recent volatility (rolling std of session ranges)
            df.loc[mask, 'volatility_5_sessions'] = symbol_data['session_range'].shift(1).rolling(5).std()
            df.loc[mask, 'volatility_10_sessions'] = symbol_data['session_range'].shift(1).rolling(10).std()

            # Volatility ratio (current vs average)
            avg_range = symbol_data['session_range'].shift(1).rolling(10).mean()
            df.loc[mask, 'range_vs_avg'] = symbol_data['session_range'].shift(1) / (avg_range + 1e-10)

            # Is volatility expanding or contracting?
            df.loc[mask, 'volatility_expanding'] = (
                    symbol_data['session_range'].shift(1).rolling(3).mean() >
                    symbol_data['session_range'].shift(1).rolling(10).mean()
            ).astype(int)

        return df

    def add_time_features(self, df):
        """
        Time-based patterns (day of week, week of month, etc.)
        Monday behaves differently from Friday!
        """
        df = df.copy()

        # Day of week (0=Monday, 4=Friday)
        df['day_of_week'] = df['session_datetime'].dt.dayofweek

        # Is it Monday or Friday? (start/end of week often different)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Week of month (1st week vs last week often different)
        df['week_of_month'] = (df['session_datetime'].dt.day - 1) // 7 + 1

        # Month (some months have different patterns)
        df['month'] = df['session_datetime'].dt.month

        return df

    def add_session_type_features(self, df):
        """
        Features specific to session type (London vs NY).
        """
        df = df.copy()

        # Binary indicator
        df['is_ny_session'] = (df['session'] == 'NY').astype(int)

        return df

    def add_relative_strength_features(self, df):
        """
        Compare current session to recent sessions.
        Is this session stronger/weaker than usual?
        """
        df = df.copy()
        df = df.sort_values(['symbol', 'session_datetime'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Z-score of current session body vs recent sessions
            body_mean = symbol_data['session_body_pct'].shift(1).rolling(10).mean()
            body_std = symbol_data['session_body_pct'].shift(1).rolling(10).std()
            df.loc[mask, 'body_zscore'] = (symbol_data['session_body_pct'].shift(1) - body_mean) / (body_std + 1e-10)

            # Percentile rank of session range
            df.loc[mask, 'range_percentile'] = symbol_data['session_range'].shift(1).rolling(20).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
            )

        return df

    def add_price_level_features(self, df):
        """
        Where is price relative to recent levels?
        Near highs? Near lows? Middle?
        """
        df = df.copy()
        df = df.sort_values(['symbol', 'session_datetime'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            # Rolling high/low over last 10 sessions
            rolling_high = symbol_data['session_high'].shift(1).rolling(10).max()
            rolling_low = symbol_data['session_low'].shift(1).rolling(10).min()

            # Where is current open relative to recent range? (0=at low, 1=at high)
            df.loc[mask, 'price_position'] = (
                    (symbol_data['session_open'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )

        return df

    def engineer_all_features(self, session_data):
        """
        Master function: Apply ALL feature engineering steps.

        Args:
            session_data: DataFrame from SessionDataBuilder

        Returns:
            DataFrame with all features added
        """
        logger.info("=" * 60)
        logger.info("ENGINEERING SESSION FEATURES")
        logger.info("=" * 60)

        df = session_data.copy()

        # Ensure we have the datetime column
        if 'session_datetime' not in df.columns:
            df['session_hour'] = df['session'].map({'London': 8, 'NY': 13})
            df['session_datetime'] = pd.to_datetime(df['session_date']) + pd.to_timedelta(df['session_hour'], unit='h')
            df.drop(columns=['session_hour'], inplace=True)

        logger.info("Adding lag features (previous sessions)...")
        df = self.add_lag_features(df)

        logger.info("Adding Asian/overnight features...")
        df = self.add_asian_session_features(df)

        logger.info("Adding multi-session trend features...")
        df = self.add_multi_session_trends(df)

        logger.info("Adding volatility features...")
        df = self.add_volatility_features(df)

        logger.info("Adding time-based features...")
        df = self.add_time_features(df)

        logger.info("Adding session type features...")
        df = self.add_session_type_features(df)

        logger.info("Adding relative strength features...")
        df = self.add_relative_strength_features(df)

        logger.info("Adding price level features...")
        df = self.add_price_level_features(df)

        # Drop rows with NaN (from rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)

        logger.info(f"\nDropped {dropped_rows} rows with NaN (from rolling windows)")
        logger.info(f"Final dataset: {len(df)} sessions")

        # Feature summary
        feature_cols = [col for col in df.columns if col not in [
            'symbol', 'session_date', 'session', 'session_datetime', 'session_id',
            'session_open', 'session_high', 'session_low', 'session_close',
            'session_volume', 'session_range', 'session_body', 'session_body_pct',
            'target'
        ]]

        logger.info(f"\n✓ Created {len(feature_cols)} features:")
        for i, col in enumerate(feature_cols, 1):
            logger.info(f"  {i}. {col}")

        return df


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Load session data from File 1
    session_data = pd.read_csv(SESSION_DATA)

    session_data['session_datetime'] = pd.to_datetime(session_data['session_datetime'])

    # Engineer features
    engineer = SessionFeatureEngineer()
    featured_data = engineer.engineer_all_features(session_data)

    # Save
    featured_data.to_csv(FEATURE_SESSION_DATA, index=False)

    print("\n✓ Feature engineering complete!")
    print(f"Shape: {featured_data.shape}")
    print(f"\nTarget distribution:")
    print(featured_data['target'].value_counts(normalize=True))

    print(f"\nSample features:")
    print(featured_data[['symbol', 'session', 'prev_session_target',
                         'last_3_sessions_bullish', 'volatility_5_sessions',
                         'is_monday', 'target']].head(10))