import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="KNN Prediction App", layout="centered")

st.title("📊 KNN Purchase Prediction App")
st.write("Predict whether a user will **Purchase** or **Not Purchase**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    knn_model, scaler = load_files()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- USER INPUT ----------------
age = st.number_input("Enter Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=10000, max_value=200000, value=50000)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = knn_model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Prediction: **Purchased**")
        st.balloons()
    else:
        st.warning("❌ Prediction: **Not Purchased**")
      
