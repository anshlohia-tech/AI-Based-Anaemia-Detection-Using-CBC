import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    TrainerConfig,
    OptimizerConfig,
)

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "anemia.csv.csv"
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ft_transformer_model"
)

ENCODER_PATH = os.path.join(
    MODELS_DIR,
    "ft_label_encoder.pkl"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(df.head())
print("\nDataset Shape :", df.shape)

# ==========================================================
# LABEL ENCODING
# ==========================================================

TARGET = "Diagnosis"

label_encoder = LabelEncoder()

df[TARGET] = label_encoder.fit_transform(df[TARGET])

df[TARGET] = df[TARGET].astype("int64")

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print("\nLabel Encoder Saved Successfully.")

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTraining Samples :", len(train_df))
print("Testing Samples  :", len(test_df))

# ==========================================================
# CONTINUOUS FEATURES
# ==========================================================

continuous_cols = [
    col for col in df.columns
    if col != TARGET
]

# ==========================================================
# DATA CONFIG
# ==========================================================

data_config = DataConfig(
    target=[TARGET],
    continuous_cols=continuous_cols,
    num_workers=0,
)

# ==========================================================
# TRAINER CONFIG
# ==========================================================

trainer_config = TrainerConfig(
    accelerator="cpu",
    batch_size=64,
    max_epochs=100,
    auto_lr_find=False,
    checkpoints=None,
    early_stopping=None,
    deterministic=True,
)

# ==========================================================
# OPTIMIZER CONFIG
# ==========================================================

optimizer_config = OptimizerConfig()

# ==========================================================
# FT TRANSFORMER CONFIG
# ==========================================================

model_config = FTTransformerConfig(

    task="classification",

    learning_rate=1e-3,

    input_embed_dim=32,

    num_heads=8,

    num_attn_blocks=4,

    transformer_head_dim=32,

    attn_dropout=0.20,

    add_norm_dropout=0.20,

    ff_dropout=0.20,

    ff_hidden_multiplier=4,

    transformer_activation="GEGLU",
)

# ==========================================================
# CREATE MODEL
# ==========================================================

print("\nCreating FT-Transformer Model...\n")

tabular_model = TabularModel(

    data_config=data_config,

    model_config=model_config,

    optimizer_config=optimizer_config,

    trainer_config=trainer_config,
)
# ==========================================================
# TRAIN MODEL
# ==========================================================

print("=" * 60)
print("Training FT-Transformer")
print("=" * 60)

import traceback

try:

    tabular_model.fit(
    train=train_df,
)

except Exception:

    print("\nTraining Failed!\n")

    traceback.print_exc()

    raise

# ==========================================================
# PREDICTION
# ==========================================================

print("\nMaking Predictions...")

prediction = tabular_model.predict(test_df)

print(prediction.head())

# ==========================================================
# TRUE LABELS
# ==========================================================

y_true = test_df[TARGET].values

# ==========================================================
# PREDICTED LABELS
# ==========================================================

if "prediction" in prediction.columns:

    y_pred = prediction["prediction"].values

elif TARGET + "_prediction" in prediction.columns:

    y_pred = prediction[TARGET + "_prediction"].values

else:

    print("\nPrediction column not found!")
    print(prediction.columns)
    raise SystemExit

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("\n")
print("=" * 60)
print("FT-TRANSFORMER RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)

# ==========================================================
# SAVE MODEL
# ==========================================================

print("\nSaving Model...")

tabular_model.save_model(MODEL_PATH)

print("\nModel Saved Successfully!")

print(MODEL_PATH)

print("\nLabel Encoder Saved At:")

print(ENCODER_PATH)

# ==========================================================
# TEST SAVED MODEL
# ==========================================================

print("\nTesting Saved Model...")

loaded_model = TabularModel.load_model(MODEL_PATH)

sample = test_df.iloc[:5]

sample_prediction = loaded_model.predict(sample)

print("\nSample Prediction")

print(sample_prediction)

print("\n")
print("=" * 60)
print("FT-Transformer Training Completed Successfully")
print("=" * 60)