import os
import sys
import joblib
import torch

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular import TabularModel

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_DIR)

from gcn.gcn_model import GCN

# ==========================================================
# LABEL ENCODERS
# ==========================================================

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

ft_label_encoder = joblib.load(
    "models/ft_label_encoder.pkl"
)

tabnet_label_encoder = joblib.load(
    "models/tabnet_label_encoder.pkl"
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

random_forest = joblib.load(
    "models/random_forest.pkl"
)

# ==========================================================
# XGBOOST
# ==========================================================

xgboost = joblib.load(
    "models/xgboost.pkl"
)

# ==========================================================
# LIGHTGBM
# ==========================================================

lightgbm = joblib.load(
    "models/lightgbm.pkl"
)

# ==========================================================
# CATBOOST
# ==========================================================

catboost = joblib.load(
    "models/catboost.pkl"
)

# ==========================================================
# MLP
# ==========================================================

mlp = joblib.load(
    "models/mlp.pkl"
)

# ==========================================================
# TABNET
# ==========================================================

tabnet = TabNetClassifier()

tabnet.load_model(
    "models/tabnet_model.zip"
)

tabnet_scaler = joblib.load(
    "models/tabnet_scaler.pkl"
)

# ==========================================================
# FT TRANSFORMER
# ==========================================================

try:

    ft_transformer = TabularModel.load_model(
        "models/ft_transformer_model"
    )

except Exception:

    ft_transformer = None

# ==========================================================
# GCN
# ==========================================================

checkpoint = torch.load(
    "models/gcn_model.pth",
    map_location="cpu"
)

gcn = GCN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_classes=checkpoint["num_classes"]
)

gcn.load_state_dict(
    checkpoint["model_state_dict"]
)

gcn.eval()

# ==========================================================
# ALL MODELS
# ==========================================================

MODELS = {

    "Random Forest": random_forest,

    "XGBoost": xgboost,

    "LightGBM": lightgbm,

    "CatBoost": catboost,

    "MLP": mlp,

    "TabNet": tabnet,

    "FT Transformer": ft_transformer,

    "GCN": gcn

}