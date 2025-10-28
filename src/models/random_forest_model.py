import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from typing import Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.rf_model = None

    def _split_X_Y(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates the DataFrame into feature matrix (X) and target vector (Y)."""
        X = data.drop(columns=['Target_Y'])
        Y = data['Target_Y']
        return X, Y

    def time_series_split(self, X: pd.DataFrame, Y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into chronological training and testing sets based on self.test_size.
        """
        split_point = int(len(X) * (1 - self.test_size))

        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        Y_train = Y.iloc[:split_point]
        Y_test = Y.iloc[split_point:]

        return X_train, X_test, Y_train, Y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Fits MinMaxScaler on training data and transforms both."""

        X_train.replace([np.inf,-np.inf],np.nan,inplace=True)
        X_test.replace([np.inf,-np.inf],np.nan,inplace=True)

        nan_indices_train = X_train.index[X_train.isnull().any(axis=1)]
        if not nan_indices_train.empty:
            X_train.drop(nan_indices_train,inplace=True)

        nan_indices_test= X_test.index[X_test.isnull().any(axis=1)]
        if not nan_indices_test.empty:
            X_test.drop(nan_indices_test, inplace=True)

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_scaled, X_test_scaled

    def balance_data(self, X: pd.DataFrame, Y: pd.Series):
        """Undersamples the majority class (0) to balance the training dataset."""

        # Combine X and Y into a single DataFrame
        combined = X.copy()
        combined['Target_Y'] = Y.values  # Use .values to avoid index issues

        # Get class counts
        class_counts = combined['Target_Y'].value_counts()
        min_class_count = class_counts.min()

        # Sample each class to have the same number of samples as the minority class
        balanced_dfs = []
        for class_label in combined['Target_Y'].unique():
            class_df = combined[combined['Target_Y'] == class_label]
            sampled_df = class_df.sample(n=min_class_count, random_state=self.random_state)
            balanced_dfs.append(sampled_df)

        # Concatenate all balanced classes
        balanced_combined = pd.concat(balanced_dfs, ignore_index=True)

        # Shuffle the data
        balanced_combined = balanced_combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Split back into X and Y
        X_resampled = balanced_combined.drop(columns=['Target_Y'])
        Y_resampled = balanced_combined['Target_Y']

        return X_resampled, Y_resampled

    def create_sequences(self, X: pd.DataFrame, Y: pd.Series, timesteps: int) -> Tuple[
        np.ndarray, np.ndarray, pd.Series]:
        """
        Converts the 2D feature DataFrame into 3D sequences (Samples, Timesteps, Features)
        required for LSTM, and adjusts the target vector accordingly.
        """
        X_array = X.values
        Y_array = Y.values

        X_seq, Y_seq, Index_seq = [], [], []

        for i in range(timesteps, len(X_array)):
            X_seq.append(X_array[i - timesteps:i, :])
            Y_seq.append(Y_array[i])
            Index_seq.append(X.index[i])

        X_final = np.array(X_seq)
        Y_final = np.array(Y_seq)
        Index_final = pd.Series(Index_seq)

        return X_final, Y_final, Index_final


    def train_random_forest(self, X_train: pd.DataFrame, Y_train: pd.Series):
        """Trains a Random Forest Classifier."""
        self.rf_model = RandomForestClassifier(n_estimators=300,max_depth=25,min_samples_leaf=3, random_state=self.random_state,
                                               n_jobs=-1)  # Use class_weight for initial tuning
        self.rf_model.fit(X_train, Y_train)
        return self.rf_model

    def get_feature_importance(self, X_data: pd.DataFrame):
        """Extracts and sorts feature importance scores from the trained RF model."""
        if self.rf_model is None:
            raise ValueError("RF model must be trained before checking importance.")

        importances = self.rf_model.feature_importances_
        feature_names = X_data.columns

        feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        print("\n--- Feature Importance Ranking ---")
        print(feature_series)
        print("----------------------------------")

        return feature_series

    def evaluate_model(self, X_test: pd.DataFrame, Y_test: pd.Series):
        """Evaluates the model on the unseen test set and focuses on Precision."""
        Y_pred = self.rf_model.predict(X_test)

        print("\n--- Model Evaluation (Test Set) ---")
        # Focus on Class 1 (UP) and Class 2 (DOWN) since Class 0 is not tradable
        print(classification_report(Y_test, Y_pred, target_names=['Neutral (0)', 'UP (1)', 'DOWN (2)']))

        # Calculate overall precision for tradable signals (1 and 2)
        tradable_precision = precision_score(Y_test, Y_pred, average='weighted', labels=[1, 2], zero_division=0)
        print(f"TRADABLE SIGNAL PRECISION (Classes 1 & 2): {tradable_precision:.4f}")

    def run_rf_pipeline(self, final_df: pd.DataFrame):
        """Runs the complete RF training and evaluation pipeline."""

        # 1. Split X and Y
        X, Y = self._split_X_Y(final_df)

        X.replace([np.inf,-np.inf],np.nan,inplace=True)

        nan_indices = X.index[X.isnull().any(axis=1)]
        if not nan_indices.empty:
            X.drop(nan_indices,inplace=True)
            Y.drop(nan_indices,inplace=True)
            logger.warning(f"Dropped {len(nan_indices)} rows containing NaN")

        # 2. Split into Training and Testing Sets (80/20)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )

        # 3. Scale Features (fit on train, transform on both)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # 4. Balance Training Data (Crucial for learning)
        X_train_res, Y_train_res = self.balance_data(X_train_scaled, Y_train)

        # 5. Train Random Forest Model
        print("\n--- Training Random Forest Model ---")
        self.train_random_forest(X_train_res, Y_train_res)

        # 6. Get Feature Importance (Diagnostic)
        self.get_feature_importance(X_train_res)

        # 7. Evaluate on Unseen Test Data
        self.evaluate_model(X_test_scaled, Y_test)

        # Return the feature importance for further use
        return X_test_scaled, Y_test, self.rf_model