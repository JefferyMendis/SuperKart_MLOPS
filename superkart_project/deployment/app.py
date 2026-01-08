import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# Load pre-trained SuperKart model from Hugging Face Model Hub
# ---------------------------------------------------------

HF_REPO_ID = "JefferyMendis/superkart-sales-model"
MODEL_FILENAME = "model_prod.joblib"

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    layout="centered"
)

st.title("🛒 SuperKart Sales Forecasting")
st.write(
    "Predict **expected sales revenue** for a product across different SuperKart stores "
    "using a production-trained machine learning model."
)

st.write("Loading trained model...")

try:
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN")  # optional but recommended
    )
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Streamlit Input UI
# ---------------------------------------------------------

st.subheader("📦 Product Information")

product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

product_allocated_area = st.slider(
    "Allocated Display Area Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

product_mrp = st.number_input(
    "Product MRP (₹)",
    min_value=1.0,
    max_value=10000.0,
    value=100.0
)

product_sugar = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_type = st.selectbox(
    "Product Type",
    [
        "Dairy", "Soft Drinks", "Snack Foods", "Frozen Foods",
        "Fruits and Vegetables", "Household", "Baking Goods",
        "Health and Hygiene", "Meat", "Seafood", "Others"
    ]
)

st.subheader("🏬 Store Information")

store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

store_city_type = st.selectbox(
    "Store City Tier",
    ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"]
)

store_est_year = st.number_input(
    "Store Establishment Year",
    min_value=1980,
    max_value=2025,
    value=2010
)

# ---------------------------------------------------------
# Prepare input DataFrame (MUST match training schema)
# ---------------------------------------------------------

input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Product_Sugar_Content": product_sugar,
    "Product_Type": product_type,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city_type,
    "Store_Type": store_type,
    "Store_Establishment_Year": store_est_year,
}])

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.subheader("📈 Sales Forecast")

if st.button("🔮 Predict Sales Revenue"):
    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 **Estimated Sales Revenue:** ₹ {prediction:,.2f}"
        )

        st.caption(
            "This prediction represents the expected total sales revenue "
            "for the selected product–store combination."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
