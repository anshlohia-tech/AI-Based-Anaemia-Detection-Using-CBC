import streamlit as st
import pandas as pd
import plotly.express as px

from utils.predict import final_prediction

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Compare Machine Learning Models")

st.write(
    """
    Select any two or more models and compare
    their predictions, confidence scores,
    severity and recommendations.
    """
)
# ---------------------------------------
# CBC INPUT
# ---------------------------------------

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
# ---------------------------------------
# MODEL SELECTION
# ---------------------------------------

st.header("Select Models")

selected_models = st.multiselect(

    "Choose Models",

    [

        "Random Forest",

        "XGBoost",

        "LightGBM",

        "CatBoost",

        "TabNet",

        "FT Transformer",

        "MLP",

        "GCN"

    ],

    default=[
        "Random Forest",
        "XGBoost"
    ]

)
compare = st.button(
    "🚀 Compare Models",
    use_container_width=True
)
# ---------------------------------------
# MODEL COMPARISON
# ---------------------------------------

if compare:

     if len(selected_models) == 0:
         st.warning("Please select at least one model.")

else:

        results = []

        with st.spinner("Comparing models..."):

            for model_name in selected_models:

                try:

                    result = final_prediction(
                        model_name,
                        input_df
                    )

                    results.append({

                        "Model": model_name,

                        "Prediction": result["disease"],

                        "Confidence (%)": result.get("confidence", 0),

                        "Severity": result["severity"],

                       "Recommendation":
                        ", ".join(result.get("recommendation", []))

                    })

                except Exception as e:

                    st.error(f"{model_name}: {e}")

        comparison_df = pd.DataFrame(results)
                # ---------------------------------------
        # COMPARISON TABLE
        # ---------------------------------------

        st.success("Comparison Completed Successfully")

        st.subheader("Comparison Results")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
                # ---------------------------------------
        # BEST MODEL
        # ---------------------------------------

        best_model = comparison_df.loc[
            comparison_df["Confidence (%)"].idxmax()
        ]

        st.success(

            f"""
🏆 Best Model

Model : {best_model['Model']}

Prediction : {best_model['Prediction']}

Confidence : {best_model['Confidence (%)']:.2f}%
            """

        )
                # ---------------------------------------
        # CONFIDENCE BAR CHART
        # ---------------------------------------

        st.subheader("📊 Confidence Comparison")

        fig = px.bar(

            comparison_df,

            x="Model",

            y="Confidence (%)",

            color="Confidence (%)",

            text="Confidence (%)",

            title="Model Confidence Comparison"

        )

        fig.update_traces(

            texttemplate="%{text:.2f}%",
            textposition="outside"

        )

        fig.update_layout(

            xaxis_title="Models",
            yaxis_title="Confidence (%)",
            height=500

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
                # ---------------------------------------
        # PREDICTION DISTRIBUTION
        # ---------------------------------------

        st.subheader("🥧 Prediction Distribution")

        pie = px.pie(

            comparison_df,

            names="Prediction",

            title="Prediction Distribution"

        )

        st.plotly_chart(

            pie,

            use_container_width=True

        )
          
                # ---------------------------------------
        # CONFIDENCE SCORE
        # ---------------------------------------

        st.subheader("📈 Confidence Score")

        for _, row in comparison_df.iterrows():

            st.write(
                f"**{row['Model']}**"
            )

            st.progress(
                min(int(row["Confidence (%)"]), 100)
            )

            st.write(
                f"Prediction : {row['Prediction']}"
            )

            st.write(
                f"Confidence : {row['Confidence (%)']:.2f}%"
            )

            st.write(
                f"Severity : {row['Severity']}"
            )

            st.divider()
                    # ---------------------------------------
        # DOWNLOAD RESULT
        # ---------------------------------------

        st.subheader("⬇ Download Results")

        csv = comparison_df.to_csv(index=False)

        st.download_button(

            label="Download Comparison CSV",

            data=csv,

            file_name="model_comparison.csv",

            mime="text/csv"

        )