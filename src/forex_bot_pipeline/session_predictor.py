import pandas as pd
import numpy as np
import pickle
from datetime import datetime, time
import logging
from src.data_pipeline.forex_cl_data import DataProcessor
from src.data_pipeline.forex_sessions import SessionDataBuilder
from src.featuress.forex_features import SessionFeatureEngineer
from src.utils.paths import DB_PATH
from src.utils.paths import MODEL_PATH,SCALER_PATH,FEATURE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionPredictor:
    """
    Real-time session prediction engine.
    Loads trained model and predicts session direction with confidence.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load trained model, scaler, and feature names."""
        try:
            model_path = MODEL_PATH
            scaler_path = SCALER_PATH
            features_path = FEATURE_PATH

            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)

            logger.info(f"✓ Model loaded successfully from {model_path}")
            logger.info(f"✓ Using {len(self.feature_names)} features")

        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise

    def determine_next_session(self, current_time=None):
        """
        Determine which session is coming up next.

        Args:
            current_time: datetime object (defaults to now)

        Returns:
            tuple: (session_name, session_start_time)
        """
        if current_time is None:
            current_time = datetime.utcnow()

        current_hour = current_time.hour

        # London: 8:00-12:00 GMT
        # NY: 13:00-17:00 GMT

        if current_hour < 8:
            return ('London', current_time.replace(hour=8, minute=0, second=0))
        elif 8 <= current_hour < 12:
            return ('London', current_time)  # Currently in London session
        elif 12 <= current_hour < 13:
            return ('NY', current_time.replace(hour=13, minute=0, second=0))
        elif 13 <= current_hour < 17:
            return ('NY', current_time)  # Currently in NY session
        else:
            # After NY closes, next is London tomorrow
            next_day = current_time.replace(hour=8, minute=0, second=0)
            return ('London', next_day)

    def get_recent_session_data(self, pair, data_processor, lookback_sessions=20):
        """
        Get recent session data for feature calculation.

        Args:
            pair: Forex pair symbol
            data_processor: Your DataProcessor instance
            lookback_sessions: How many recent sessions to fetch

        Returns:
            DataFrame with recent session data
        """
        # Load recent data from database
        hourly_data = data_processor.load_forex_data([pair])

        if hourly_data.empty:
            logger.warning(f"No data available for {pair}")
            return pd.DataFrame()

        builder = SessionDataBuilder(data_processor)
        engineer = SessionFeatureEngineer()

        # Assign sessions
        hourly_data['symbol'] = pair
        hourly_with_sessions = builder.assign_session(hourly_data)

        # Create session OHLC
        session_data = builder.create_session_ohlc(hourly_with_sessions)

        if session_data.empty:
            return pd.DataFrame()

        # Add sequence ID
        session_data = builder.add_session_sequence_id(session_data)

        # Engineer features
        featured_data = engineer.engineer_all_features(session_data)

        # Get last N sessions
        recent_sessions = featured_data.tail(lookback_sessions)

        return recent_sessions

    def prepare_prediction_features(self, recent_sessions, next_session_type):
        """
        Prepare features for the NEXT session prediction.

        Args:
            recent_sessions: DataFrame with recent session data
            next_session_type: 'London' or 'NY'

        Returns:
            Feature vector ready for prediction
        """
        if recent_sessions.empty or len(recent_sessions) < 10:
            logger.warning("Not enough recent data for reliable prediction")
            return None

        # Get the most recent session as baseline
        last_session = recent_sessions.iloc[-1]

        # Create feature dict (matching training features)
        features = {}

        # Copy all feature values from last session
        for feature in self.feature_names:
            if feature in last_session:
                features[feature] = last_session[feature]
            else:
                features[feature] = 0  # Default if missing

        # Update session type indicator
        features['is_ny_session'] = 1 if next_session_type == 'NY' else 0
        features['is_london'] = 1 if next_session_type == 'London' else 0

        # Create DataFrame with single row
        feature_df = pd.DataFrame([features])

        # Ensure column order matches training
        feature_df = feature_df[self.feature_names]

        return feature_df

    def predict_session(self, pair, next_session_type, data_processor):
        """
        Make prediction for the next session.

        Args:
            pair: Forex pair symbol
            next_session_type: 'London' or 'NY'
            data_processor: DataProcessor instance

        Returns:
            dict: {
                'pair': str,
                'session': str,
                'direction': 'BULLISH' or 'BEARISH',
                'confidence': float (0-1),
                'prediction_time': datetime
            }
        """
        try:
            # Get recent session data
            recent_sessions = self.get_recent_session_data(pair, data_processor)

            if recent_sessions.empty:
                logger.warning(f"Cannot predict {pair} - no recent data")
                return None

            # Prepare features
            features = self.prepare_prediction_features(recent_sessions, next_session_type)

            if features is None:
                return None

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)

            # Map prediction to direction
            direction = 'BULLISH' if prediction == 1 else 'BEARISH'

            result = {
                'pair': pair,
                'session': next_session_type,
                'direction': direction,
                'confidence': confidence,
                'prediction_time': datetime.utcnow(),
                'probabilities': {
                    'bearish': probabilities[0],
                    'bullish': probabilities[1]
                }
            }

            logger.info(f"✓ {pair} {next_session_type}: {direction} ({confidence:.1%} confidence)")

            return result

        except Exception as e:
            logger.error(f"✗ Error predicting {pair}: {e}")
            return None

    def predict_all_pairs(self, pairs, data_processor, min_confidence=0.70):
        """
        Make predictions for all pairs and filter by confidence.

        Args:
            pairs: List of forex pairs
            data_processor: DataProcessor instance
            min_confidence: Minimum confidence threshold

        Returns:
            List of prediction dicts sorted by confidence (highest first)
        """
        # Determine next session
        next_session, _ = self.determine_next_session()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PREDICTING NEXT SESSION: {next_session}")
        logger.info(f"{'=' * 60}")

        predictions = []

        for pair in pairs:
            pred = self.predict_session(pair, next_session, data_processor)

            if pred and pred['confidence'] >= min_confidence:
                predictions.append(pred)

        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"SUMMARY: {len(predictions)} pairs above {min_confidence:.0%} confidence")
        logger.info(f"{'=' * 60}")

        if predictions:
            logger.info("\nTop Signals:")
            for i, pred in enumerate(predictions[:5], 1):
                logger.info(f"  {i}. {pred['pair']:.<12} {pred['direction']:.<8} {pred['confidence']:.1%}")
        else:
            logger.info("No signals meet confidence threshold")

        return predictions


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Initialize
    db_path = str(DB_PATH)
    data_processor = DataProcessor(db_path=db_path)

    predictor = SessionPredictor()

    # Define pairs
    pairs = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
        "EURJPY=X", "GBPJPY=X", "GC=F"
    ]

    # Get predictions for next session
    predictions = predictor.predict_all_pairs(
        pairs=pairs,
        data_processor=data_processor,
        min_confidence=0.70
    )

    # Display results
    print("\n" + "=" * 60)
    print("TRADEABLE SIGNALS")
    print("=" * 60)

    for pred in predictions:
        print(f"\nPair: {pred['pair']}")
        print(f"Session: {pred['session']}")
        print(f"Direction: {pred['direction']}")
        print(f"Confidence: {pred['confidence']:.2%}")
        print(f"Probabilities: Bearish {pred['probabilities']['bearish']:.2%} | "
              f"Bullish {pred['probabilities']['bullish']:.2%}")