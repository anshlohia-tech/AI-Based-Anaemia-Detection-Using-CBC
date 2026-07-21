import streamlit as st
import pandas as pd

from utils.predict import final_prediction
st.set_page_config(
    page_title="Anemia Prediction",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Anemia Prediction System")

st.write(
    "Enter CBC values below and select a machine learning model."
)
model = st.selectbox(

    "Select Machine Learning Model",

    [

        "Random Forest",

        "XGBoost",

        "LightGBM",

        "CatBoost",

        "MLP",

        "TabNet",

        "FT Transformer",

        "GCN"

    ]

)
st.header("Enter CBC Values")

col1, col2 = st.columns(2)

with col1:

    WBC = st.number_input("WBC", value=5.0)
    LYMp = st.number_input("LYMp", value=30.0)
    NEUTp = st.number_input("NEUTp", value=60.0)
    LYMn = st.number_input("LYMn", value=2.0)
    NEUTn = st.number_input("NEUTn", value=4.0)
    RBC = st.number_input("RBC", value=4.5)
    HGB = st.number_input("HGB", value=12.0)

with col2:

    HCT = st.number_input("HCT", value=40.0)
    MCV = st.number_input("MCV", value=90.0)
    MCH = st.number_input("MCH", value=30.0)
    MCHC = st.number_input("MCHC", value=33.0)
    PLT = st.number_input("PLT", value=250.0)
    PDW = st.number_input("PDW", value=12.0)
    PCT = st.number_input("PCT", value=0.20)
    input_df = pd.DataFrame({

    "WBC":[WBC],
    "LYMp":[LYMp],
    "NEUTp":[NEUTp],
    "LYMn":[LYMn],
    "NEUTn":[NEUTn],
    "RBC":[RBC],
    "HGB":[HGB],
    "HCT":[HCT],
    "MCV":[MCV],
    "MCH":[MCH],
    "MCHC":[MCHC],
    "PLT":[PLT],
    "PDW":[PDW],
    "PCT":[PCT]

})
    predict_btn = st.button(

    "🔍 Predict",

    use_container_width=True

)
if predict_btn:

    with st.spinner("Predicting..."):

        result = final_prediction(

            model,

            input_df

        )

    st.success("Prediction Completed")

    st.subheader("Prediction Result")

    st.write(f"### Model : {model}")

    st.write(f"### Prediction : {result['disease']}")

    st.write(f"### Confidence : {result['confidence']} %")

    st.write(f"### Severity : {result['severity']}")
    
    st.subheader("Recommendations")

    for item in result["recommendation"]:

        st.write("•", item)