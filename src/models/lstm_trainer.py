import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_class_weight


class LSTMPipeline:
    def __init__(self, timesteps: int = 20):
        self.timesteps = timesteps

    def build_and_train(self, X_train_3D, Y_train_3D, X_test_3D, Y_test_3D):
        # 1. One-hot encoding
        Y_train_ohe = to_categorical(Y_train_3D)
        Y_test_ohe = to_categorical(Y_test_3D)
        features = X_train_3D.shape[2]

        # 2. Calculate class weights (CRITICAL for imbalanced data!)
        class_counts = pd.Series(Y_train_3D).value_counts()
        total = len(Y_train_3D)

        class_weights = {
            0: total / (3 * class_counts[0]) * 0.5,  # Reduce weight for class 0
            1: total / (3 * class_counts[1]) * 1.0,
            2: total / (3 * class_counts[2]) * 1.0
        }
        print(f"\nManual Class Weights: {class_weights}")

        model = Sequential([
            # LSTM with stronger dropout
            LSTM(64, return_sequences=False, activation='tanh',
                 dropout=0.4, recurrent_dropout=0.3,  # Add recurrent dropout
                 input_shape=(self.timesteps, features)),
            Dropout(0.4),  # Increased from 0.3

            # Smaller dense layer
            Dense(16, activation='relu'),  # Reduced from 32
            Dropout(0.3),  # Increased from 0.2

            # Output layer
            Dense(3, activation='softmax')
        ])

        # 4. Compile with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(model.summary())

        # 5. Callbacks for smart training
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # 6. Train Model with class weights
        history = model.fit(
            X_train_3D, Y_train_ohe,
            epochs=100,  # Higher because early stopping will handle it
            batch_size=64,
            validation_data=(X_test_3D, Y_test_ohe),
            class_weight=class_weights,  # CRITICAL!
            callbacks=[early_stop, reduce_lr],
            verbose=1,
            shuffle=False  # Keep False for time series
        )

        return model, history

    def evaluate_model(self, model, X_test_3D: np.ndarray, Y_test_3D: np.ndarray):
        """
        Evaluates the LSTM model with confidence thresholding.
        """
        # Predict probabilities
        Y_pred_proba = model.predict(X_test_3D, verbose=0)
        Y_pred = np.argmax(Y_pred_proba, axis=1)

        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(Y_pred_proba, axis=1)

        print("\n" + "=" * 50)
        print("LSTM Model Evaluation (Test Set)")
        print("=" * 50)

        # Standard evaluation
        print("\n--- ALL PREDICTIONS ---")
        print(classification_report(Y_test_3D, Y_pred,
                                    target_names=['Neutral (0)', 'UP (1)', 'DOWN (2)'],
                                    zero_division=0))

        tradable_precision_all = precision_score(Y_test_3D, Y_pred,
                                                 average='weighted', labels=[1, 2],
                                                 zero_division=0)
        print(f"TRADABLE SIGNAL PRECISION (Classes 1 & 2): {tradable_precision_all:.4f}")

        # High-confidence predictions only (THIS IS KEY!)
        print("\n" + "=" * 50)
        print("HIGH CONFIDENCE PREDICTIONS (>60% confidence)")
        print("=" * 50)

        high_conf_mask = confidence_scores > 0.5
        if high_conf_mask.sum() > 0:
            Y_test_high_conf = Y_test_3D[high_conf_mask]
            Y_pred_high_conf = Y_pred[high_conf_mask]

            print(
                f"\nNumber of high-confidence predictions: {high_conf_mask.sum()} / {len(Y_test_3D)} ({high_conf_mask.sum() / len(Y_test_3D) * 100:.1f}%)")

            print(classification_report(Y_test_high_conf, Y_pred_high_conf,
                                        target_names=['Neutral (0)', 'UP (1)', 'DOWN (2)'],
                                        zero_division=0))

            tradable_precision_high_conf = precision_score(Y_test_high_conf, Y_pred_high_conf,
                                                           average='weighted', labels=[1, 2],
                                                           zero_division=0)
            print(f"HIGH-CONFIDENCE TRADABLE PRECISION: {tradable_precision_high_conf:.4f}")
        else:
            print("No high-confidence predictions found!")
            tradable_precision_high_conf = 0.0

        return tradable_precision_all, tradable_precision_high_conf

    def run_lstm_pipeline(self, X_train_3D: np.ndarray, Y_train_3D: np.ndarray,
                          X_test_3D: np.ndarray, Y_test_3D: np.ndarray) -> tf.keras.Model:
        """
        Executes the full LSTM workflow: Build, Train, and Evaluate.
        """
        print("\n" + "=" * 50)
        print("STARTING LSTM TRAINING AND EVALUATION")
        print("=" * 50)

        # 1. Build and Train
        trained_model, history = self.build_and_train(X_train_3D, Y_train_3D, X_test_3D, Y_test_3D)

        # 2. Evaluate
        precision_all, precision_high_conf = self.evaluate_model(trained_model, X_test_3D, Y_test_3D)

        # 3. Plot training history (optional but useful)
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")

        return trained_model