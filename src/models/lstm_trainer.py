import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, precision_score


class LSTMPipeline:
    def __init__(self, timesteps: int = 20):
        self.timesteps = timesteps

    def build_and_train(self, X_train_3D, Y_train_3D, X_test_3D, Y_test_3D):
        # 1. Encoding
        Y_train_ohe = to_categorical(Y_train_3D)
        Y_test_ohe = to_categorical(Y_test_3D)
        features = X_train_3D.shape[2]  # Get the number of features

        # 2. Build Model (as described above)
        model = Sequential([
            LSTM(64, input_shape=(self.timesteps, features), activation='tanh'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        # 3. Train Model
        history = model.fit(
            X_train_3D, Y_train_ohe,
            epochs=30,  # Start with 30 epochs
            batch_size=64,
            validation_data=(X_test_3D, Y_test_ohe),
            verbose=1,
            shuffle=False
        )

        return model

    def evaluate_model(self, model, X_test_3D: np.ndarray, Y_test_3D: np.ndarray):
        """
        Evaluates the LSTM model and prints the classification report and tradable precision.
        """

        # Predict probabilities
        Y_pred_proba = model.predict(X_test_3D, verbose=0)
        # Convert probabilities back to class labels (0, 1, or 2)
        Y_pred = np.argmax(Y_pred_proba, axis=1)

        print("\n--- LSTM Model Evaluation (Test Set) ---")

        # Classification report provides detailed metrics for all classes
        print(classification_report(Y_test_3D, Y_pred,
                                    target_names=['Neutral (0)', 'UP (1)', 'DOWN (2)'],
                                    zero_division=0))

        # Calculate overall precision for tradable signals (1 and 2)
        # We use 'weighted' average only on labels 1 and 2 to get the combined precision
        tradable_precision = precision_score(Y_test_3D, Y_pred,
                                             average='weighted', labels=[1, 2], zero_division=0)
        print(f"TRADABLE SIGNAL PRECISION (Classes 1 & 2): {tradable_precision:.4f}")

        return tradable_precision


    def run_lstm_pipeline(self, X_train_3D: np.ndarray, Y_train_3D: np.ndarray,
                          X_test_3D: np.ndarray, Y_test_3D: np.ndarray) -> tf.keras.Model:
        """
        Executes the full LSTM workflow: Build, Train, and Evaluate.
        Returns the trained Keras model.
        """
        print("\n" + "=" * 40)
        print("STARTING LSTM TRAINING AND EVALUATION")
        print("=" * 40)

        # 1. Build and Train the model (Using the previously defined logic)
        trained_model = self.build_and_train(X_train_3D, Y_train_3D, X_test_3D, Y_test_3D)

        # 2. Evaluate the model
        self.evaluate_model(trained_model, X_test_3D, Y_test_3D)

        return trained_model