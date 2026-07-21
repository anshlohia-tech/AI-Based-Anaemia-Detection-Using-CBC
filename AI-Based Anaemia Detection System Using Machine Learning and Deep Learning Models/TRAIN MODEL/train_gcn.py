import os
import sys
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Add project folder to Python path
sys.path.append(PROJECT_DIR)

# Import GCN files
from gcn.graph_utils import load_graph_data
from gcn.gcn_model import GCN

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "gcn_model.pth")
ENCODER_PATH = os.path.join(MODEL_DIR, "gcn_label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "gcn_metrics.pkl")

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "anemia.csv.csv"
)

# ==========================================================
# LOAD GRAPH DATA
# ==========================================================

print("=" * 60)
print("Loading Graph Dataset")
print("=" * 60)

X, y, A, encoder = load_graph_data(DATASET_PATH)

print(f"Samples : {len(y)}")
print(f"Features: {X.shape[1]}")
print(f"Classes : {len(torch.unique(y))}")

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

indices = torch.arange(len(y))

train_idx, test_idx = train_test_split(
    indices.numpy(),
    test_size=0.20,
    random_state=42,
    stratify=y.numpy()
)

train_idx = torch.tensor(train_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

print(f"\nTraining Samples : {len(train_idx)}")
print(f"Testing Samples  : {len(test_idx)}")

# ==========================================================
# CREATE MODEL
# ==========================================================

model = GCN(
    input_dim=X.shape[1],
    hidden_dim=32,
    num_classes=len(torch.unique(y))
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

# ==========================================================
# TRAINING
# ==========================================================

epochs = 200

print("\n" + "=" * 60)
print("Training GCN")
print("=" * 60)

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    output = model(X, A)

    loss = criterion(
        output[train_idx],
        y[train_idx]
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        train_prediction = output.argmax(dim=1)

        train_accuracy = (
            train_prediction[train_idx] == y[train_idx]
        ).float().mean().item()

        print(
            f"Epoch {epoch+1:3d}/{epochs}"
            f" | Loss = {loss.item():.4f}"
            f" | Train Accuracy = {train_accuracy:.4f}"
        )

# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating Model...")

model.eval()

with torch.no_grad():

    output = model(X, A)

    prediction = output.argmax(dim=1)

pred = prediction[test_idx].cpu().numpy()
true = y[test_idx].cpu().numpy()

accuracy = accuracy_score(true, pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    true,
    pred,
    average="weighted",
    zero_division=0
)

print("\n")
print("=" * 60)
print("GCN RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        true,
        pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# ==========================================================
# SAVE MODEL
# ==========================================================

print("\nSaving Model...")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden_dim": 32,
        "num_classes": len(torch.unique(y))
    },
    MODEL_PATH
)

joblib.dump(
    encoder,
    ENCODER_PATH
)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}

joblib.dump(
    metrics,
    METRICS_PATH
)

print("Model Saved Successfully!")

print(MODEL_PATH)

print("\nLabel Encoder Saved At:")

print(ENCODER_PATH)

print("\nMetrics Saved At:")

print(METRICS_PATH)

# ==========================================================
# TEST SAVED MODEL
# ==========================================================

print("\nTesting Saved Model...")

checkpoint = torch.load(MODEL_PATH)

loaded_model = GCN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_classes=checkpoint["num_classes"]
)

loaded_model.load_state_dict(
    checkpoint["model_state_dict"]
)

loaded_model.eval()

with torch.no_grad():
    sample_prediction = loaded_model(X, A).argmax(dim=1)

sample_prediction = sample_prediction[test_idx].cpu().numpy()

print("\nFirst Five Predictions")

for i in range(5):

    print(
        f"Sample {i+1}"
        f" | True = {encoder.inverse_transform([true[i]])[0]}"
        f" | Predicted = {encoder.inverse_transform([sample_prediction[i]])[0]}"
    )

print("\n" + "=" * 60)
print("GCN Training Completed Successfully")
print("=" * 60)