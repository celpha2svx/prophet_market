import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import logging
from src.utils.paths import FEATURE_SESSION_DATA,MODEL_PATH,SCALER_PATH,FEATURE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionModelTrainer:
    """
    Trains Random Forest to predict session direction (Bullish/Bearish).
    Evaluates by confidence levels to find optimal trading threshold.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None

    def prepare_data(self, df):
        """
        Prepare features and target, split train/test chronologically.
        """
        logger.info("Preparing data for training...")

        # Define feature columns (exclude metadata and target)
        exclude_cols = [
            'symbol', 'session_date', 'session', 'session_datetime', 'session_id',
            'session_open', 'session_high', 'session_low', 'session_close',
            'session_volume', 'session_range', 'session_body', 'session_body_pct',
            'target'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols

        logger.info(f"Using {len(feature_cols)} features")

        # Sort by datetime for chronological split
        df = df.sort_values('session_datetime')

        X = df[feature_cols]
        y = df['target']
        metadata = df[['symbol', 'session_date', 'session', 'session_datetime']]

        # Chronological split (last 20% for testing)
        split_idx = int(len(df) * (1 - self.test_size))
        split_date = df['session_datetime'].iloc[split_idx]

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        metadata_test = metadata.iloc[split_idx:]

        logger.info(f"\nSplit at: {split_date}")
        logger.info(f"Train: {len(X_train)} sessions ({df['session_datetime'].min()} to {split_date})")
        logger.info(f"Test: {len(X_test)} sessions ({split_date} to {df['session_datetime'].max()})")

        logger.info(f"\nTrain target distribution:")
        print(y_train.value_counts(normalize=True))
        logger.info(f"\nTest target distribution:")
        print(y_test.value_counts(normalize=True))

        return X_train, X_test, y_train, y_test, metadata_test

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        logger.info("\nScaling features...")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        """Train Random Forest classifier."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("=" * 60)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle any residual imbalance
        )

        self.model.fit(X_train, y_train)

        logger.info("✓ Model trained successfully")

        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train).mean()
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        return self.model

    def get_feature_importance(self, X_train):
        """Get and display feature importance."""
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        logger.info("\n" + "=" * 60)
        logger.info("TOP 15 MOST IMPORTANT FEATURES")
        logger.info("=" * 60)
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"{row['feature']:.<40} {row['importance']:.4f}")

        return feature_importance

    def evaluate_model(self, X_test, y_test, metadata_test):
        """Comprehensive evaluation with confidence-based analysis."""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)

        # Get predictions and probabilities
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Confidence score = max probability
        confidence = np.max(y_pred_proba, axis=1)

        # Overall performance
        logger.info("\n--- ALL PREDICTIONS ---")
        logger.info(classification_report(y_test, y_pred,
                                          target_names=['Bearish (0)', 'Bullish (1)'],
                                          digits=4))

        overall_accuracy = (y_pred == y_test).mean()
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"              Predicted Bearish  Predicted Bullish")
        logger.info(f"Actual Bearish      {cm[0, 0]:<15}  {cm[0, 1]}")
        logger.info(f"Actual Bullish      {cm[1, 0]:<15}  {cm[1, 1]}")

        # Performance by confidence threshold
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE BY CONFIDENCE THRESHOLD")
        logger.info("=" * 60)

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        results = []

        for thresh in thresholds:
            mask = confidence >= thresh
            if mask.sum() > 0:
                high_conf_pred = y_pred[mask]
                high_conf_true = y_test[mask]

                accuracy = (high_conf_pred == high_conf_true).mean()
                precision = precision_score(high_conf_true, high_conf_pred, zero_division=0)
                recall = recall_score(high_conf_true, high_conf_pred, zero_division=0)
                count = mask.sum()
                pct = (count / len(y_test)) * 100

                results.append({
                    'threshold': thresh,
                    'count': count,
                    'percentage': pct,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                })

                logger.info(f"\nConfidence >= {thresh:.0%}:")
                logger.info(f"  Predictions: {count:,} ({pct:.1f}% of test set)")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")

        results_df = pd.DataFrame(results)

        # Performance by session type
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE BY SESSION TYPE")
        logger.info("=" * 60)

        for session_type in ['London', 'NY']:
            session_mask = metadata_test['session'] == session_type
            if session_mask.sum() > 0:
                session_pred = y_pred[session_mask]
                session_true = y_test[session_mask]
                session_acc = (session_pred == session_true).mean()

                logger.info(f"\n{session_type} Sessions:")
                logger.info(f"  Count: {session_mask.sum()}")
                logger.info(f"  Accuracy: {session_acc:.4f}")

        # Performance by pair (top 5)
        logger.info("\n" + "=" * 60)
        logger.info("TOP 5 BEST PERFORMING PAIRS")
        logger.info("=" * 60)

        pair_performance = []
        for symbol in metadata_test['symbol'].unique():
            pair_mask = metadata_test['symbol'] == symbol
            if pair_mask.sum() >= 20:  # Only if enough samples
                pair_pred = y_pred[pair_mask]
                pair_true = y_test[pair_mask]
                pair_acc = (pair_pred == pair_true).mean()
                pair_performance.append({
                    'pair': symbol,
                    'accuracy': pair_acc,
                    'count': pair_mask.sum()
                })

        pair_df = pd.DataFrame(pair_performance).sort_values('accuracy', ascending=False)
        for idx, row in pair_df.head(5).iterrows():
            logger.info(f"{row['pair']:.<15} Accuracy: {row['accuracy']:.4f} (n={row['count']})")

        # Create evaluation report
        eval_report = {
            'overall_accuracy': overall_accuracy,
            'confidence_thresholds': results_df,
            'pair_performance': pair_df,
            'test_size': len(y_test),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidence': confidence,
            'y_true': y_test,
            'metadata': metadata_test
        }

        return eval_report

    def save_model(self):
        """Save trained model, scaler, and feature names."""
        model_path = MODEL_PATH
        scaler_path = SCALER_PATH
        features_path = FEATURE_PATH

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)

        logger.info(f"\n✓ Model saved to {model_path}")
        logger.info(f"✓ Scaler saved to {scaler_path}")
        logger.info(f"✓ Features saved to {features_path}")

    def train_pipeline(self, featured_data, save_model=True):
        """
        Complete training pipeline.
        """
        logger.info("=" * 60)
        logger.info("SESSION MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        # 1. Prepare data
        X_train, X_test, y_train, y_test, metadata_test = self.prepare_data(featured_data)

        # 2. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # 3. Train model
        self.train_model(X_train_scaled, y_train)

        # 4. Feature importance
        feature_importance = self.get_feature_importance(X_train_scaled)

        # 5. Evaluate
        eval_report = self.evaluate_model(X_test_scaled, y_test, metadata_test)

        # 6. Save model
        if save_model:
            self.save_model()

        logger.info("\n" + "=" * 60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("=" * 60)

        return eval_report, feature_importance


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Load featured data from File 2
    featured_data = pd.read_csv(FEATURE_SESSION_DATA)

    # Parse datetime
    featured_data['session_datetime'] = pd.to_datetime(featured_data['session_datetime'])

    # Train model
    trainer = SessionModelTrainer(test_size=0.2, random_state=42)
    eval_report, feature_importance = trainer.train_pipeline(featured_data, save_model=True)

    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {eval_report['overall_accuracy']:.2%}")
    print("\nAccuracy by Confidence Threshold:")
    print(eval_report['confidence_thresholds'][['threshold', 'accuracy', 'percentage']])

    print("\n✓ Model ready for deployment!")