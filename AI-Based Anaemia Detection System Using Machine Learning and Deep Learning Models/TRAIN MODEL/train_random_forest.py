import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("dataset/anemia.csv.csv")

# Features and Target
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

# Save Model
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Random Forest Model Saved Successfully.")