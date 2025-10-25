import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from src.utils.paths import ALL_FEATURE_CSV, ALL_CLEANED_CSV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    Cleans feature data for ML models by handling NaN values intelligently.
    """

    def __init__(self):
        self.cleaning_report = {}

    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns per stock and feature.
        """
        report = {}

        for symbol in df['symbol'].unique():
            stock_data = df[df['symbol'] == symbol]

            missing_info = {}
            for col in stock_data.columns:
                if col in ['symbol', 'date']:
                    continue

                null_count = stock_data[col].isna().sum()
                null_pct = (null_count / len(stock_data)) * 100

                if null_count > 0:
                    missing_info[col] = {
                        'count': null_count,
                        'percentage': round(null_pct, 2),
                        'first_valid_idx': stock_data[col].first_valid_index()
                    }

            if missing_info:
                report[symbol] = missing_info

        return report

    def clean_stock_data(self, stock_df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Clean data for a single stock.

        Strategies:
        - 'drop': Drop all rows with any NaN
        - 'forward_fill': Forward fill NaN values
        - 'smart': Drop leading NaNs, forward fill remaining
        """
        stock_df = stock_df.copy().sort_values('date')

        if strategy == 'drop':
            # Simple: drop all rows with any NaN
            cleaned = stock_df.dropna()
            logger.info(f"Dropped {len(stock_df) - len(cleaned)} rows with NaN")

        elif strategy == 'forward_fill':
            # Forward fill all NaN values
            cleaned = stock_df.fillna(method='ffill')
            # If still NaN at start, backward fill
            cleaned = cleaned.fillna(method='bfill')

        elif strategy == 'smart':
            # Smart approach:
            # 1. Drop leading NaNs (first X rows that don't have all features)
            # 2. Keep data from first row with all features complete

            # Find first row where ALL feature columns have values
            feature_cols = [col for col in stock_df.columns if col not in ['symbol', 'date']]

            # Check each row to see if it has all features
            complete_rows = stock_df[feature_cols].notna().all(axis=1)

            if complete_rows.any():
                first_complete_idx = complete_rows.idxmax()
                cleaned = stock_df.loc[first_complete_idx:].copy()

                # Forward fill any remaining isolated NaNs
                cleaned[feature_cols] = cleaned[feature_cols].fillna(method='ffill')

                logger.info(f"Kept {len(cleaned)} rows (dropped {len(stock_df) - len(cleaned)} leading NaN rows)")
            else:
                # No complete rows - use forward fill approach
                logger.warning(f"No complete rows found, using forward fill")
                cleaned = stock_df.fillna(method='ffill').fillna(method='bfill')

        return cleaned

    def clean_all_stocks(self, df: pd.DataFrame, strategy: str = 'smart', min_rows: int = 200) -> pd.DataFrame:
        """
        Clean data for all stocks in the dataset.

        Args:
            df: Input DataFrame with all stocks
            strategy: Cleaning strategy to use
            min_rows: Minimum rows required per stock after cleaning
        """
        logger.info("Starting data cleaning pipeline...")

        # Analyze missing data first
        missing_report = self.analyze_missing_data(df)

        if missing_report:
            logger.info(f"Found missing data in {len(missing_report)} stocks")
            for symbol, info in list(missing_report.items())[:5]:  # Show first 5
                logger.info(f"  {symbol}: {len(info)} columns with NaN")

        cleaned_stocks = []
        stocks_removed = []

        for symbol in df['symbol'].unique():
            stock_data = df[df['symbol'] == symbol].copy()

            # Clean the stock data
            cleaned_stock = self.clean_stock_data(stock_data, strategy=strategy)

            # Check if enough data remains
            if len(cleaned_stock) < min_rows:
                logger.warning(
                    f"❌ {symbol}: Only {len(cleaned_stock)} rows after cleaning (min: {min_rows}). REMOVING!")
                stocks_removed.append(symbol)
            else:
                cleaned_stocks.append(cleaned_stock)
                logger.info(f"✅ {symbol}: {len(cleaned_stock)} clean rows")

        # Combine all cleaned stocks
        if cleaned_stocks:
            cleaned_df = pd.concat(cleaned_stocks, ignore_index=True)
        else:
            cleaned_df = pd.DataFrame()

        # Generate cleaning report
        self.cleaning_report = {
            'original_stocks': df['symbol'].nunique(),
            'cleaned_stocks': len(cleaned_stocks),
            'removed_stocks': stocks_removed,
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'rows_dropped': len(df) - len(cleaned_df),
            'strategy_used': strategy
        }

        return cleaned_df

    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate that cleaned data is ready for ML.
        """
        validation = {
            'has_nulls': df.isnull().any().any(),
            'null_columns': df.columns[df.isnull().any()].tolist(),
            'total_rows': len(df),
            'total_stocks': df['symbol'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        }

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validation['has_inf'] = np.isinf(df[numeric_cols]).any().any()

        if validation['has_inf']:
            validation['inf_columns'] = df[numeric_cols].columns[np.isinf(df[numeric_cols]).any()].tolist()

        return validation

    def print_cleaning_summary(self):
        """
        Print summary of cleaning operation.
        """
        print("\n" + "=" * 60)
        print("DATA CLEANING SUMMARY")
        print("=" * 60)

        report = self.cleaning_report
        print(f"Original stocks: {report['original_stocks']}")
        print(f"Cleaned stocks: {report['cleaned_stocks']}")
        print(f"Removed stocks: {len(report['removed_stocks'])}")

        if report['removed_stocks']:
            print(f"  Removed: {', '.join(report['removed_stocks'])}")

        print(f"\nOriginal rows: {report['original_rows']:,}")
        print(f"Cleaned rows: {report['cleaned_rows']:,}")
        print(
            f"Rows dropped: {report['rows_dropped']:,} ({(report['rows_dropped'] / report['original_rows'] * 100):.1f}%)")
        print(f"\nStrategy used: {report['strategy_used']}")


# ============ EXAMPLE USAGE ============

def main():
    print("Loading raw features data...")
    df = pd.read_csv(ALL_FEATURE_CSV)
    df['date'] = pd.to_datetime(df['date'])

    print(f"\nOriginal data: {len(df):,} rows, {df['symbol'].nunique()} stocks")

    # Initialize cleaning pipeline
    cleaner = DataCleaningPipeline()

    # Analyze missing data
    print("\n" + "=" * 60)
    print("ANALYZING MISSING DATA")
    print("=" * 60)
    missing_report = cleaner.analyze_missing_data(df)

    if missing_report:
        print(f"\nStocks with missing data: {len(missing_report)}")

        # Show detailed report for first 3 stocks
        for symbol in list(missing_report.keys())[:3]:
            print(f"\n{symbol}:")
            for col, info in missing_report[symbol].items():
                print(f"  {col}: {info['count']} NaN ({info['percentage']}%)")
    else:
        print("✅ No missing data found!")

    # Clean the data
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    cleaned_df = cleaner.clean_all_stocks(
        df,
        strategy='smart',  # Try: 'drop', 'forward_fill', 'smart'
        min_rows=200  # Minimum 200 days of clean data per stock
    )

    # Print cleaning summary
    cleaner.print_cleaning_summary()

    # Validate cleaned data
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    validation = cleaner.validate_cleaned_data(cleaned_df)

    print(f"Has NaN values: {validation['has_nulls']}")
    if validation['has_nulls']:
        print(f"Columns with NaN: {validation['null_columns']}")
    else:
        print("✅ No NaN values!")

    print(f"\nHas infinite values: {validation['has_inf']}")
    if validation['has_inf']:
        print(f"Columns with inf: {validation['inf_columns']}")

    print(f"\nFinal dataset:")
    print(f"  Total rows: {validation['total_rows']:,}")
    print(f"  Total stocks: {validation['total_stocks']}")
    print(f"  Date range: {validation['date_range']['start'].date()} to {validation['date_range']['end'].date()}")
    print(f"  Total days: {validation['date_range']['days']}")

    # Show sample of cleaned data
    print("\n" + "=" * 60)
    print("SAMPLE OF CLEANED DATA")
    print("=" * 60)
    print(cleaned_df.head(10))

    # Check for any remaining issues
    print("\n" + "=" * 60)
    print("COLUMN-BY-COLUMN CHECK")
    print("=" * 60)
    for col in cleaned_df.columns:
        null_count = cleaned_df[col].isnull().sum()
        print(f"{col:20s}: {null_count:5d} nulls")

    # Save cleaned data
    cleaned_df.to_csv(ALL_CLEANED_CSV, index=False)
    print(f"\n✅ Cleaned data saved to: {ALL_CLEANED_CSV}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("ROWS PER STOCK (after cleaning)")
    print("=" * 60)
    rows_per_stock = cleaned_df.groupby('symbol').size().sort_values(ascending=False)
    print(rows_per_stock.head(20))
    print(f"\nMin rows: {rows_per_stock.min()}")
    print(f"Max rows: {rows_per_stock.max()}")
    print(f"Average rows: {rows_per_stock.mean():.0f}")