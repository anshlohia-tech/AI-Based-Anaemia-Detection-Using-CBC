import streamlit as st

st.set_page_config(
    page_title="Anaemia AI",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Anaemia AI Prediction System")

st.markdown("---")

st.header("Welcome")

st.write("""
This project predicts different types of anaemia using Complete Blood Count (CBC) parameters
and a Machine Learning model.

### Features

✅ CBC Analysis

✅ Random Forest Prediction

✅ Explainable AI

✅ Diet Recommendation

✅ Treatment Suggestion

✅ PDF Report Generation

---

### Navigation

Use the left sidebar to open:

- Prediction
- About
- Diet
- Treatment
- Report
""")

st.success("Model Status : Ready")