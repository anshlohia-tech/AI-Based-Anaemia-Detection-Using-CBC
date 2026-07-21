import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------
# Paths
# ------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "anemia.csv.csv"
)

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv(DATA_PATH)

print(df.head())

# ------------------------------------
# Features and Target
# ------------------------------------

X = df.drop("Diagnosis", axis=1)

y = df["Diagnosis"]

# ------------------------------------
# Encode Labels
# ------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

# ------------------------------------
# Train Test Split
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------
# Create MLP
# ------------------------------------

mlp = MLPClassifier(

    hidden_layer_sizes=(128,64),

    activation="relu",

    solver="adam",

    max_iter=500,

    random_state=42

)

# ------------------------------------
# Train
# ------------------------------------

print("Training MLP...")

mlp.fit(X_train, y_train)

# ------------------------------------
# Accuracy
# ------------------------------------

pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"Accuracy : {accuracy:.4f}")

# ------------------------------------
# Save Model
# ------------------------------------

joblib.dump(

    mlp,

    os.path.join(MODEL_DIR, "mlp.pkl")

)

print("MLP Saved Successfully.")