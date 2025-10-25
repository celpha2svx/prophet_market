import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import logging
import json
from datetime import timedelta
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockForecaster:
    """
    Time series forecasting for stock prices using ARIMA.
    Includes walk-forward validation and directional accuracy metrics.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.best_order = None
        self.forecast_history = []
        self.metrics = {}

    def prepare_data(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Extract and prepare time series data for the symbol.
        """
        stock_data = features_df[features_df['symbol'] == self.symbol].copy()
        stock_data = stock_data.sort_values('date')

        # Use close price as target
        ts = stock_data.set_index('date')['close']

        logger.info(f"Prepared {len(ts)} data points for {self.symbol}")
        return ts

    def find_best_arima_order(self, ts: pd.Series, max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Grid search for best ARIMA(p,d,q) parameters using AIC.
        Returns best order tuple.
        """
        best_aic = np.inf
        best_order = None

        logger.info(f"Searching for best ARIMA parameters for {self.symbol}...")

        # Try different combinations
        for p in range(0, max_p + 1):
            for d in range(0, 2):  # Usually 0 or 1 for stock prices
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        logger.info(f"Best ARIMA order for {self.symbol}: {best_order} (AIC: {best_aic:.2f})")
        return best_order

    def train(self, ts: pd.Series, order: Tuple[int, int, int] = None):
        """
        Train ARIMA model on time series data.
        """
        if order is None:
            # Auto-select best parameters
            order = self.find_best_arima_order(ts)

        self.best_order = order

        try:
            self.model = ARIMA(ts, order=order)
            self.fitted_model = self.model.fit()
            logger.info(f"Model trained successfully for {self.symbol}")
        except Exception as e:
            logger.error(f"Error training model for {self.symbol}: {e}")
            raise

    def forecast(self, steps: int = 5) -> pd.DataFrame:
        """
        Generate forecast for next N steps.
        Returns DataFrame with predictions and confidence intervals.
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet!")

        forecast_result = self.fitted_model.forecast(steps=steps)

        # Get confidence intervals
        forecast_df = self.fitted_model.get_forecast(steps=steps).summary_frame()

        forecast_df['predicted_price'] = forecast_result
        forecast_df = forecast_df.rename(columns={
            'mean': 'forecast',
            'mean_ci_lower': 'lower_bound',
            'mean_ci_upper': 'upper_bound'
        })

        return forecast_df[['forecast', 'lower_bound', 'upper_bound']]

    def walk_forward_validation(self, ts: pd.Series, train_size: int = 100, test_size: int = 30, step: int = 5) -> Dict:
        """
        Perform walk-forward validation to test model accuracy.

        Args:
            train_size: Initial training window size
            test_size: Number of points to validate
            step: How many steps ahead to forecast each time
        """
        logger.info(f"Starting walk-forward validation for {self.symbol}...")

        predictions = []
        actuals = []
        dates = []

        for i in range(train_size, len(ts) - step, step):
            # Split data
            train_data = ts.iloc[:i]
            test_data = ts.iloc[i:i + step]

            try:
                # Train model
                if self.best_order is None:
                    # Use simple default for speed in validation
                    model = ARIMA(train_data, order=(1, 1, 1))
                else:
                    model = ARIMA(train_data, order=self.best_order)

                fitted = model.fit()

                # Forecast
                forecast = fitted.forecast(steps=len(test_data))

                # Store results
                for j, (pred, actual, date) in enumerate(zip(forecast, test_data.values, test_data.index)):
                    predictions.append(pred)
                    actuals.append(actual)
                    dates.append(date)

            except Exception as e:
                logger.warning(f"Validation error at step {i}: {e}")
                continue

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Directional accuracy
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100

        metrics = {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'directional_accuracy': round(directional_accuracy, 2),
            'num_predictions': len(predictions)
        }

        self.metrics = metrics

        logger.info(f"Validation complete for {self.symbol}")
        logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Direction: {directional_accuracy:.2f}%")

        # Store for plotting
        self.validation_results = pd.DataFrame({
            'date': dates,
            'actual': actuals,
            'predicted': predictions
        })

        return metrics

    def plot_validation(self, save_path: str = None):
        """
        Plot walk-forward validation results.
        """
        if not hasattr(self, 'validation_results'):
            logger.warning("No validation results to plot")
            return

        df = self.validation_results

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Actual vs Predicted
        ax1 = axes[0]
        ax1.plot(df['date'], df['actual'], label='Actual Price', color='blue', linewidth=2)
        ax1.plot(df['date'], df['predicted'], label='Predicted Price', color='red', linestyle='--', linewidth=2)
        ax1.set_title(f'{self.symbol} - Walk-Forward Validation: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction Error
        ax2 = axes[1]
        errors = df['actual'] - df['predicted']
        ax2.plot(df['date'], errors, label='Prediction Error', color='green', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.fill_between(df['date'], 0, errors, alpha=0.3, color='green')
        ax2.set_title(f'{self.symbol} - Prediction Error Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Add metrics text
        metrics_text = f"MAE: ${self.metrics['mae']:.2f} | RMSE: ${self.metrics['rmse']:.2f} | MAPE: {self.metrics['mape']:.1f}% | Direction Accuracy: {self.metrics['directional_accuracy']:.1f}%"
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plot saved to {save_path}")

        plt.show()

    def plot_forecast(self, ts: pd.Series, forecast_df: pd.DataFrame, save_path: str = None):
        """
        Plot historical data with forecast.
        """
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot historical data (last 90 days)
        recent_ts = ts.tail(90)
        ax.plot(recent_ts.index, recent_ts.values, label='Historical Price', color='blue', linewidth=2)

        # Create future dates for forecast
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast_df), freq='D')

        # Plot forecast
        ax.plot(future_dates, forecast_df['forecast'], label='Forecast', color='red', linewidth=2, linestyle='--')

        # Plot confidence interval
        ax.fill_between(future_dates,
                        forecast_df['lower_bound'],
                        forecast_df['upper_bound'],
                        alpha=0.3, color='red', label='95% Confidence Interval')

        ax.set_title(f'{self.symbol} - Price Forecast (Next 5 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")

        plt.show()

    def save_model_metrics(self, filename: str = 'forecast_metrics.json'):
        """
        Save model metrics to JSON file.
        """
        model_info = {
            'symbol': self.symbol,
            'model_type': 'ARIMA',
            'order': self.best_order,
            'metrics': self.metrics
        }

        # Load existing metrics if file exists
        try:
            with open(filename, 'r') as f:
                all_metrics = json.load(f)
        except FileNotFoundError:
            all_metrics = {}

        # Update with this model's metrics
        all_metrics[self.symbol] = model_info

        # Save
        with open(filename, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Metrics saved to {filename}")


# ============ EXAMPLE USAGE ============

def main():
    # Load features
    print("Loading features...")
    features_df = pd.read_csv('../../Data/files_csv/all_features_cleaned.csv')
    features_df['date'] = pd.to_datetime(features_df['date'])

    # Select stocks to forecast (start with 2-3)
    symbols_to_forecast = ['AAPL', 'MSFT', 'WMT']

    all_forecasts = {}

    for symbol in symbols_to_forecast:
        print("\n" + "=" * 60)
        print(f"FORECASTING: {symbol}")
        print("=" * 60)

        # Initialize forecaster
        forecaster = StockForecaster(symbol)

        # Prepare data
        ts = forecaster.prepare_data(features_df)

        if len(ts) < 50:
            print(f"Insufficient data for {symbol}, skipping...")
            continue

        # Walk-forward validation
        print(f"\nRunning walk-forward validation for {symbol}...")
        metrics = forecaster.walk_forward_validation(ts, train_size=80, test_size=20, step=5)

        print(f"\nValidation Metrics:")
        print(f"  MAE: ${metrics['mae']:.2f}")
        print(f"  RMSE: ${metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

        # Plot validation results
        forecaster.plot_validation(save_path=f'{symbol}_validation.png')

        # Train on full dataset
        print(f"\nTraining final model for {symbol}...")
        forecaster.train(ts, order=(1, 1, 1))  # Using simple order for speed

        # Generate 5-day forecast
        print(f"\nGenerating 5-day forecast for {symbol}...")
        forecast_df = forecaster.forecast(steps=5)

        print(f"\nForecast Results:")
        print(forecast_df)

        # Plot forecast
        forecaster.plot_forecast(ts, forecast_df, save_path=f'{symbol}_forecast.png')

        # Save metrics
        forecaster.save_model_metrics()

        # Store forecast
        all_forecasts[symbol] = {
            'current_price': ts.iloc[-1],
            'forecast': forecast_df['forecast'].tolist(),
            'metrics': metrics
        }

    # Summary report
    print("\n" + "=" * 60)
    print("FORECASTING SUMMARY")
    print("=" * 60)

    for symbol, data in all_forecasts.items():
        current = data['current_price']
        day5_forecast = data['forecast'][-1]
        change = ((day5_forecast - current) / current) * 100
        direction = "UP" if change > 0 else "DOWN"

        print(f"\n{symbol}:")
        print(f"  Current Price: ${current:.2f}")
        print(f"  5-Day Forecast: ${day5_forecast:.2f}")
        print(f"  Expected Change: {change:+.2f}% ({direction})")
        print(f"  Model Accuracy: {data['metrics']['directional_accuracy']:.1f}%")
        print(f"  MAPE: {data['metrics']['mape']:.2f}%")

    print("\n✅ Forecasting complete! Check the generated PNG files for visualizations.")