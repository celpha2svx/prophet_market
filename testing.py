from src.data_pipeline.forex_cl_data import DataProcessor
from src.featuress.forex_features import FeatureEngineer
from src.models.random_forest_model import ModelTrainer
from src.models.lstm_trainer import LSTMPipeline
from src.utils.paths import  DB_PATH
import pandas as pd
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TIMESTEPS = 20

db_path = str(DB_PATH)
pipeline = DataProcessor(db_path=db_path)

print("\n" + "=" * 50)
print("Calculating features for multiple stocks...")
features =["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
           "NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X","GC=F"]

all_data = pipeline.create_target(features)
final_features = []

# Iterate over each unique symbol in the combined data
for symbol in all_data['symbol'].unique():
    symbol_data = all_data[all_data['symbol'] == symbol].copy()

    # Instantiate FeatureEngineer for the specific symbol
    fe = FeatureEngineer(symbol=symbol)

    # Add features
    processed_data = fe.add_features(symbol_data)

    final_features.append(processed_data)

# Combine all final feature sets for the model
X_combined = pd.concat(final_features, ignore_index=False)
NUM_FEATURES = len(X_combined.columns) - 1 # Target_Y is 1 column

# --- 2. PREPROCESSING (SPLIT, SCALE, BALANCE, SEQUENCE) ---
logger.info("--- STEP 2: Data Preprocessing for LSTM ---")
model_trainer = ModelTrainer(test_size=0.2, random_state=42)

# Get X and Y
X, Y = model_trainer._split_X_Y(X_combined)

# Split (2D)
X_train_2D, X_test_2D, Y_train_1D, Y_test_1D = model_trainer.time_series_split(X, Y)

# Scale (2D)
X_train_scaled, X_test_scaled = model_trainer.scale_features(X_train_2D, X_test_2D)

# CRITICAL: Get the indices that survived scaling (after NaN drops)
train_indices_kept = X_train_scaled.index
test_indices_kept = X_test_scaled.index

# Filter Y to match the kept indices
Y_train_aligned = Y_train_1D[Y_train_1D.index.isin(train_indices_kept)]
Y_test_aligned = Y_test_1D[Y_test_1D.index.isin(test_indices_kept)]

# Sort both X and Y by index to ensure they're in the same order
X_train_scaled = X_train_scaled.sort_index()
Y_train_aligned = Y_train_aligned.sort_index()
X_test_scaled = X_test_scaled.sort_index()
Y_test_aligned = Y_test_aligned.sort_index()

# Now reset indices to be clean and sequential
X_train_clean = X_train_scaled.reset_index(drop=True)
Y_train_clean = Y_train_aligned.reset_index(drop=True)
X_test_clean = X_test_scaled.reset_index(drop=True)
Y_test_clean = Y_test_aligned.reset_index

print(f"X_train_scaled min: {X_train_scaled.min().min()}")
print(f"X_train_scaled max: {X_train_scaled.max().max()}")
print(f"X_train_scaled has inf: {np.isinf(X_train_scaled.values).any()}")

print(f"X_train_clean shape: {X_train_clean.shape}")
print(f"Y_train_clean shape: {Y_train_clean.shape}")
print(f"X_train_clean length: {len(X_train_clean)}")
print(f"Y_train_clean length: {len(Y_train_clean)}")

# Balance Training Data (2D)
X_train_res, Y_train_res = model_trainer.balance_data(X_train_clean,Y_train_clean)

# Convert to Sequences (3D)
X_train_3D, Y_train_3D, _ = model_trainer.create_sequences(X_train_res, Y_train_res, TIMESTEPS)
X_test_3D, Y_test_3D, X_test_index = model_trainer.create_sequences(X_test_scaled, Y_test_1D, TIMESTEPS)

logger.info(f"LSTM Train Shape: {X_train_3D.shape}, Test Shape: {X_test_3D.shape}")

# Add this right before calling lstm_pipeline.run_lstm_pipeline()
print("\n=== DATA DIAGNOSTICS ===")
print(f"X_train shape: {X_train_3D.shape}")
print(f"Y_train distribution:\n{pd.Series(Y_train_3D).value_counts()}")
print(f"\nX_train stats:")
print(f"  Min: {X_train_3D.min():.4f}")
print(f"  Max: {X_train_3D.max():.4f}")
print(f"  Mean: {X_train_3D.mean():.4f}")
print(f"  Has NaN: {np.isnan(X_train_3D).any()}")
print(f"  Has Inf: {np.isinf(X_train_3D).any()}")

# --- 3. LSTM TRAINING AND EVALUATION ---
logger.info("--- STEP 3: LSTM Training and Evaluation ---")
lstm_pipeline = LSTMPipeline(timesteps=TIMESTEPS)

# Execute the full training and evaluation
lstm_model = lstm_pipeline.run_lstm_pipeline(
    X_train_3D,
    Y_train_3D,
    X_test_3D,
    Y_test_3D
)

logger.info("Pipeline execution complete. Check report above.")