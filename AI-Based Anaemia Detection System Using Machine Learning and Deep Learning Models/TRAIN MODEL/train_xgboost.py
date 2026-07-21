import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/anemia.csv.csv")

# ==========================
# Features & Target
# ==========================

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# ==========================
# Encode Labels
# ==========================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save Label Encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Train Model
# ==========================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# ==========================
# Evaluate
# ==========================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"Accuracy : {accuracy*100:.2f}%")

# ==========================
# Save Model
# ==========================

joblib.dump(model, "models/xgboost.pkl")

print("XGBoost Model Saved Successfully.")