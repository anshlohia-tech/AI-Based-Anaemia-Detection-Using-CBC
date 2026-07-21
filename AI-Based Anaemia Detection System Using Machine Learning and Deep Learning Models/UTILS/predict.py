import numpy as np
import torch

from utils.load_models import *

# -----------------------------------------------------
# Generic Prediction (Scikit-Learn Models)
# -----------------------------------------------------

def sklearn_predict(model, df, encoder):

    prediction = model.predict(df)

    prediction = np.array(prediction).astype(int)

    disease = encoder.inverse_transform(prediction)[0]

    confidence = 100.0

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(df)

        confidence = float(probability.max() * 100)

    return disease, confidence


# -----------------------------------------------------
# Random Forest
# -----------------------------------------------------

def predict_random_forest(df):

    return sklearn_predict(
        random_forest,
        df,
        label_encoder
    )


# -----------------------------------------------------
# XGBoost
# -----------------------------------------------------

def predict_xgboost(df):

    return sklearn_predict(
        xgboost,
        df,
        label_encoder
    )


# -----------------------------------------------------
# LightGBM
# -----------------------------------------------------

def predict_lightgbm(df):

    return sklearn_predict(
        lightgbm,
        df,
        label_encoder
    )


# -----------------------------------------------------
# CatBoost
# -----------------------------------------------------

def predict_catboost(df):

    return sklearn_predict(
        catboost,
        df,
        label_encoder
    )


# -----------------------------------------------------
# MLP
# -----------------------------------------------------

def predict_mlp(df):

    return sklearn_predict(
        mlp,
        df,
        label_encoder
    )


# -----------------------------------------------------
# TabNet
# -----------------------------------------------------

def predict_tabnet(df):

    X = tabnet_scaler.transform(df)

    prediction = tabnet.predict(X)

    probability = tabnet.predict_proba(X)

    disease = tabnet_label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

    confidence = float(probability.max() * 100)

    return disease, confidence


# -----------------------------------------------------
# FT Transformer
# -----------------------------------------------------

def predict_ft_transformer(df):

    if ft_transformer is None:
        return "FT Transformer Not Loaded", 0.0

    prediction = ft_transformer.predict(df)

    # Prediction class
    pred = prediction["Diagnosis_prediction"].values[0]

    disease = ft_label_encoder.inverse_transform(
        np.array([pred]).astype(int)
    )[0]

    # All probability columns
    prob_cols = [
        c for c in prediction.columns
        if "_probability" in c
    ]

    confidence = (
        prediction[prob_cols]
        .iloc[0]
        .max()
        * 100
    )

    return disease, confidence

# -----------------------------------------------------
# GCN
# -----------------------------------------------------

def predict_gcn(df):

    X = torch.tensor(
        df.values,
        dtype=torch.float32
    )

    A = torch.eye(len(df))

    with torch.no_grad():

        output = gcn(X, A)

        probability = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probability,
            dim=1
        )

    disease = label_encoder.inverse_transform(
        prediction.numpy()
    )[0]

    confidence = float(
        probability.max().item() * 100
    )

    return disease, confidence


# -----------------------------------------------------
# MASTER PREDICT FUNCTION
# -----------------------------------------------------

def predict(model_name, df):

    if model_name == "Random Forest":

        return predict_random_forest(df)

    elif model_name == "XGBoost":

        return predict_xgboost(df)

    elif model_name == "LightGBM":

        return predict_lightgbm(df)

    elif model_name == "CatBoost":

        return predict_catboost(df)

    elif model_name == "MLP":

        return predict_mlp(df)

    elif model_name == "TabNet":

        return predict_tabnet(df)

    elif model_name == "FT Transformer":

        return predict_ft_transformer(df)

    elif model_name == "GCN":

        return predict_gcn(df)

    else:

        return "Unknown Model", 0.0

# -----------------------------------------------------
# ANEMIA SEVERITY
# -----------------------------------------------------

def anemia_severity(disease, confidence):

    disease = disease.lower()

    if disease in ["healthy", "normal", "no anemia"]:

        return "Normal"

    if confidence >= 85:

        return "High Risk"

    elif confidence >= 60:

        return "Moderate Risk"

    else:

        return "Low Risk"
    
# -----------------------------------------------------
# RECOMMENDATIONS
# -----------------------------------------------------

def generate_recommendation(severity):

    recommendations = {

        "Normal": [

            "Maintain a balanced diet.",

            "Continue regular health checkups.",

            "Exercise regularly."

        ],

        "Low Risk": [

            "Increase iron-rich foods.",

            "Eat green leafy vegetables.",

            "Monitor CBC regularly."

        ],

        "Moderate Risk": [

            "Consult a physician.",

            "Increase iron and Vitamin C intake.",

            "Repeat CBC if symptoms continue."

        ],

        "High Risk": [

            "Seek immediate medical advice.",

            "Perform Complete Blood Count (CBC).",

            "Follow prescribed treatment."

        ]

    }

    return recommendations.get(

        severity,

        ["Consult a healthcare professional."]

    )

# -----------------------------------------------------
# FINAL PREDICTION
# -----------------------------------------------------

def final_prediction(model_name, df):

    disease, confidence = predict(

        model_name,

        df

    )

    severity = anemia_severity(

        disease,

        confidence

    )

    recommendation = generate_recommendation(

        severity

    )

    return {

        "disease": disease,

        "confidence": round(confidence, 2),

        "severity": severity,

        "recommendation": recommendation

    }