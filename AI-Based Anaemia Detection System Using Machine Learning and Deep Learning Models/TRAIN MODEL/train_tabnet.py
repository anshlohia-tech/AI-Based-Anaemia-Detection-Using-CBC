import os
import joblib
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from pytorch_tabnet.tab_model import TabNetClassifier 


# =====================================================
# PATHS
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "anemia.csv.csv")

MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


print("="*60)
print("Loading Dataset")
print("="*60)

df = pd.read_csv(DATASET_PATH)

print(df.head())
print("\nShape :", df.shape)


# =====================================================
# FEATURES
# =====================================================

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


# =====================================================
# LABEL ENCODER
# =====================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(
    encoder,
    os.path.join(MODELS_DIR, "tabnet_label_encoder.pkl")
)


# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    os.path.join(MODELS_DIR, "tabnet_scaler.pkl")
)

X = X.astype(np.float32)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining :", len(X_train))
print("Testing  :", len(X_test))


# =====================================================
# TABNET MODEL
# =====================================================

print("\nTraining TabNet...\n")

model = TabNetClassifier(

    n_d=64,
    n_a=64,
    n_steps=5,

    gamma=1.5,

    lambda_sparse=1e-4,

    optimizer_fn=torch.optim.Adam,

    optimizer_params=dict(
        lr=0.02
    ),

    scheduler_params={
        "step_size":50,
        "gamma":0.9
    },

    scheduler_fn=torch.optim.lr_scheduler.StepLR,

    mask_type="entmax",

    seed=42,

    verbose=10

)


model.fit(

    X_train,

    y_train,

    eval_set=[(X_test, y_test)],

    eval_name=["Validation"],

    eval_metric=["accuracy"],

    max_epochs=500,

    patience=100,

    batch_size=256,

    virtual_batch_size=128,

    drop_last=False

)


# =====================================================
# PREDICTION
# =====================================================

pred = model.predict(X_test)


# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(y_test, pred)

precision = precision_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)


print("\n")
print("="*60)
print("TABNET RESULTS")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


# =====================================================
# SAVE MODEL
# =====================================================

MODEL_PATH = os.path.join(MODELS_DIR, "tabnet_model")

model.save_model(MODEL_PATH)

print("\nTabNet Model Saved Successfully")
print(MODEL_PATH + ".zip")