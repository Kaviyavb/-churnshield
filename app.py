import streamlit as st
import pandas as pd
import pickle
import base64

# ----------------------------
# 🔹 Load model and training columns
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("training_columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

# ----------------------------
# 🔹 Streamlit page setup
# ----------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.markdown("Upload a customer CSV to predict churn. Uses a trained Random Forest model.")

# ----------------------------
# 🔹 Show sample CSV structure
# ----------------------------
st.markdown("### 🧾 Sample Input CSV Format")

try:
    with open("sample_input.csv", "r") as file:
        st.code(file.read(), language="csv")
except FileNotFoundError:
    st.warning("sample_input.csv not found in your directory.")

# ----------------------------
# 🔹 Download link for sample CSV
# ----------------------------
def download_sample(file_path, label="📥 Download sample_input.csv"):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_input.csv">{label}</a>'
        st.markdown(href, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ sample_input.csv is missing!")

download_sample("sample_input.csv")

# ----------------------------
# 🔹 File uploader
# ----------------------------
uploaded_file = st.file_uploader(" ", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Preview input
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Drop unnecessary columns
        df = df.drop(columns=["Churn"], errors='ignore')
        df = df.drop(columns=["customerID"], errors='ignore')

        # One-hot encode to match training columns
        df_encoded = pd.get_dummies(df)

        # Add missing columns
        for col in training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns to match training
        df_encoded = df_encoded[training_columns]

        # Predict
        predictions = model.predict(df_encoded)
        df["Churn Prediction"] = ["Churn" if p == 1 else "No Churn" for p in predictions]

        st.success("✅ Prediction completed!")
        st.dataframe(df)

        # Download results
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv_download, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error reading or processing file: {e}")
else:
    st.info("👆 Upload a valid customer CSV file to get started.")
