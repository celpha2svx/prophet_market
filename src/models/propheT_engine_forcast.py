import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import logging
import json
from typing import Dict, List, Tuple
from src.utils.paths import ALL_CLEANED_CSV,PROPHET_METRICS_JSON,STOCK_PREDICTABILY,IMG_DIR


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetStockForecaster:
    """
    Enhanced stock forecasting using Prophet with additional regressors.
    Includes walk-forward validation and model comparison.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.forecast_result = None
        self.validation_results = None
        self.metrics = {}

    def prepare_prophet_data(self, features_df: pd.DataFrame, include_regressors: bool = True) -> pd.DataFrame:
        """
        Prepare data in Prophet format with optional regressors.
        Prophet requires columns: 'ds' (date) and 'y' (target)
        """
        stock_data = features_df[features_df['symbol'] == self.symbol].copy()
        # Ensure 'date' is datetime and tz-naive
        stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce').dt.tz_localize(None)
        stock_data = stock_data.sort_values('date')

        # Prophet format
        prophet_df = pd.DataFrame({
            'ds': stock_data['date'],
            'y': stock_data['close']
        })

        # Add regressors (additional features to help prediction)
        if include_regressors:
            # Volume (normalized)
            # Volume (simple version - no normalization)
            if 'volume' in stock_data.columns:
                prophet_df['volume'] = stock_data['volume'].values

            # Volatility
            if 'realized_volatility' in stock_data.columns:
                prophet_df['volatility'] = stock_data['realized_volatility'].values

            # RSI (momentum indicator)
            if 'rsi' in stock_data.columns:
                prophet_df['rsi'] = stock_data['rsi'].values

            # MACD histogram (trend strength)
            if 'macd_histogram' in stock_data.columns:
                prophet_df['macd_hist'] = stock_data['macd_histogram'].values

            # Average True Range (volatility measure)
            if 'atr' in stock_data.columns:
                prophet_df['atr'] = stock_data['atr'].values

        # Remove remaining NaN
        prophet_df = prophet_df.fillna(method='ffill').fillna(method='bfill')
        logger.info(f"Prepared {len(prophet_df)} data points for {self.symbol}")
        return prophet_df

    def train(self, prophet_df: pd.DataFrame, include_regressors: bool = True):
        """
        Train Prophet model with optional regressors.
        """
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )

        if include_regressors:
            regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
            for col in regressor_cols:
                self.model.add_regressor(col)
                logger.info(f"Added regressor: {col}")

        self.model.fit(prophet_df)
        logger.info(f"Prophet model trained for {self.symbol}")

    def forecast(self, periods: int = 5, prophet_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate forecast for next N periods.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        future = self.model.make_future_dataframe(periods=periods)

        if prophet_df is not None:
            regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
            for col in regressor_cols:
                # Extend regressors with last known value
                last_values = prophet_df[col].tail(periods).mean()
                future[col] = np.full(len(future), last_values)

        forecast = self.model.predict(future)
        result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        result_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        self.forecast_result = result_df
        return result_df

    def walk_forward_validation(self, prophet_df: pd.DataFrame, train_size: int = 300,
                                  test_steps: int = 5, n_splits: int = 20,
                                  include_regressors: bool = True) -> Dict:
        """
        Perform walk-forward validation.
        """
        logger.info(f"Starting walk-forward validation for {self.symbol}...")

        predictions = []
        actuals = []
        dates = []

        total_points = len(prophet_df)
        step_size = (total_points - train_size) // n_splits

        for i in range(n_splits):
            start_idx = 0
            end_idx = train_size + (i * step_size)
            if end_idx + test_steps > total_points:
                break

            train_data = prophet_df.iloc[start_idx:end_idx]
            test_data = prophet_df.iloc[end_idx:end_idx + test_steps]

            try:
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05
                )

                if include_regressors:
                    regressor_cols = [col for col in train_data.columns if col not in ['ds', 'y']]
                    for col in regressor_cols:
                        model.add_regressor(col)

                model.fit(train_data)
                future = pd.DataFrame({'ds': test_data['ds']})
                if include_regressors:
                    for col in regressor_cols:
                        future[col] = test_data[col].values

                forecast = model.predict(future)

                for pred, actual, date in zip(forecast['yhat'].values, test_data['y'].values, test_data['ds'].values):
                    predictions.append(pred)
                    actuals.append(actual)
                    dates.append(date)

            except Exception as e:
                logger.warning(f"Validation error at split {i}: {e}")
                continue

        if len(predictions) == 0:
            logger.error(f"No successful validation runs for {self.symbol}")
            return None

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        price_accuracy_5pct = np.mean(np.abs((actuals - predictions) / actuals) < 0.05) * 100

        self.metrics = {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'directional_accuracy': round(directional_accuracy, 2),
            'price_accuracy_5pct': round(price_accuracy_5pct, 2),
            'num_predictions': len(predictions)
        }
        self.validation_results = pd.DataFrame({
            'date': dates,
            'actual': actuals,
            'predicted': predictions
        })

        logger.info(f"Validation complete for {self.symbol}")
        logger.info(f"Direction: {directional_accuracy:.2f}%, MAPE: {mape:.2f}%")
        return self.metrics

    def plot_validation(self, save_path: str = IMG_DIR):
        """Plot validation results."""
        if self.validation_results is None:
            logger.warning("No validation results to plot")
            return

        df = self.validation_results
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        # Plot 1: Actual vs Predicted
        axes[0].plot(df['date'], df['actual'], label='Actual', color='blue', linewidth=2)
        axes[0].plot(df['date'], df['predicted'], label='Predicted', color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f'{self.symbol} - Prophet Validation (Actual vs Predicted)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Plot 2: Prediction Error
        axes[1].plot(df['date'], df['actual'] - df['predicted'], color='green', linewidth=1.5)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].fill_between(df['date'], 0, df['actual'] - df['predicted'], alpha=0.3, color='green')
        axes[1].set_title(f'{self.symbol} - Prediction Error', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Error ($)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        # Add metrics text
        metrics_text = (f"MAE: ${self.metrics['mae']:.2f} | RMSE: ${self.metrics['rmse']:.2f} | "
                        f"MAPE: {self.metrics['mape']:.1f}% | Direction: {self.metrics['directional_accuracy']:.1f}%")
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, filename: str = PROPHET_METRICS_JSON):
        """Save metrics to JSON."""
        model_info = {
            'symbol': self.symbol,
            'model_type': 'Prophet',
            'metrics': self.metrics
        }
        try:
            with open(filename, 'r') as f:
                all_metrics = json.load(f)
        except FileNotFoundError:
            all_metrics = {}
        all_metrics[self.symbol] = model_info
        with open(filename, 'w') as f:
            json.dump(all_metrics, f, indent=2)


def compare_models_all_stocks(features_df: pd.DataFrame,
                              symbols: List[str] = None,
                              quick_mode: bool = False) -> pd.DataFrame:
    """
    Compare ARIMA vs Prophet for all stocks and create leaderboard.

    Args:
        quick_mode: If True, use fewer validation splits for speed
    """
    if symbols is None:
        symbols = features_df['symbol'].unique()

    results = []

    for symbol in symbols:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {symbol}")
        logger.info(f"{'=' * 60}")

        # Prophet model
        prophet_forecaster = ProphetStockForecaster(symbol)
        prophet_df = prophet_forecaster.prepare_prophet_data(features_df, include_regressors=True)

        if len(prophet_df) < 200:
            logger.warning(f"Insufficient data for {symbol}")
            continue

        # Validate
        n_splits = 10 if quick_mode else 20
        prophet_metrics = prophet_forecaster.walk_forward_validation(
            prophet_df,
            train_size=250,
            n_splits=n_splits,
            include_regressors=True
        )

        if prophet_metrics is None:
            continue

        # Save plot
        prophet_forecaster.plot_validation(f'{symbol}_prophet_validation.png')

        # Store results
        results.append({
            'symbol': symbol,
            'model': 'Prophet',
            'directional_accuracy': prophet_metrics['directional_accuracy'],
            'mape': prophet_metrics['mape'],
            'mae': prophet_metrics['mae'],
            'rmse': prophet_metrics['rmse'],
            'price_accuracy_5pct': prophet_metrics['price_accuracy_5pct']
        })

        # Save individual metrics
        prophet_forecaster.save_metrics()

    # Create leaderboard
    leaderboard = pd.DataFrame(results)

    if len(leaderboard) == 0:
        logger.error("No successful forecasts! Returning empty leaderboard.")
        return pd.DataFrame(columns=['symbol', 'model', 'directional_accuracy', 'mape', 'mae', 'rmse'])

    leaderboard = leaderboard.sort_values('directional_accuracy', ascending=False)

    return leaderboard


# Usage Example
def main():
    print("Loading cleaned features...")
    features_df = pd.read_csv(ALL_CLEANED_CSV)
    features_df['date'] = pd.to_datetime(features_df['date'])
    print(f"Loaded {len(features_df)} rows, {features_df['symbol'].nunique()} stocks")

    print("\nAvailable columns:")
    print(features_df.columns.tolist())
    print(f"\nAAPL rows before prepare: {len(features_df[features_df['symbol'] == 'AAPL'])}")

    test_symbols = ['AAPL', 'MSFT', 'WMT']
    leaderboard = compare_models_all_stocks(features_df, symbols=test_symbols, quick_mode=True)
    print(leaderboard.to_string(index=False))
    # Compare with ARIMA results
    print("\n" + "=" * 60)
    print("COMPARISON: ARIMA vs Prophet")
    print("=" * 60)
    print("\nARIMA Results (from yesterday):")
    print("AAPL: 49.86%")
    print("MSFT: 48.78%")
    print("WMT: 48.51%")

    print("\nProphet Results (with features):")
    for _, row in leaderboard.iterrows():
        print(f"{row['symbol']}: {row['directional_accuracy']:.2f}%")

    # Ask user if they want to run on all 41 stocks
    print("\n" + "=" * 60)
    print("Ready to run on all 41 stocks?")
    print("This will take 20-30 minutes...")
    print("=" * 60)

    run_all = input("Run on all 41 stocks? (yes/no): ").strip().lower()

    if run_all == 'yes':
        print("\nRunning full analysis on all 41 stocks...")
        full_leaderboard = compare_models_all_stocks(
            features_df,
            quick_mode=False  # Full validation
        )

        print("\n" + "=" * 60)
        print("FULL LEADERBOARD - All 41 Stocks")
        print("=" * 60)
        print(full_leaderboard.to_string(index=False))

        # Save leaderboard
        full_leaderboard.to_csv(STOCK_PREDICTABILY, index=False)
        print("\n✅ Leaderboard saved to: stock_predictability_leaderboard.csv")

        # Find Golden stocks (>52% accuracy)
        golden_stocks = full_leaderboard[full_leaderboard['directional_accuracy'] > 52]
        print(f"\n🏆 GOLDEN STOCKS (>52% accuracy): {len(golden_stocks)}")
        print(golden_stocks[['symbol', 'directional_accuracy', 'mape']].to_string(index=False))

    print("\n✅ Prophet forecasting complete!")