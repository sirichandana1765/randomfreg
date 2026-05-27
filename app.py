import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RF Regression", page_icon="🌲", layout="wide")

# =========================
# BACKGROUND STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b, #334155);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/random_model.pkl")
scaler = joblib.load("models/scaler.pkl")

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["target"] = housing.target

# =========================
# TITLE
# =========================
st.title("🌲 Random Forest Regression Dashboard")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Enter Features")

inputs = []
for feature in housing.feature_names:
    val = st.sidebar.number_input(feature, value=0.0)
    inputs.append(val)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("Predict 💰"):

    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"🏠 Predicted House Price: {prediction[0]:.2f}")

# =========================
# VISUALIZATIONS
# =========================

st.markdown("## 📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["target"], kde=True, ax=ax)
    st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================

st.markdown("## 🌟 Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": housing.feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots()
ax.barh(feat_df["Feature"], feat_df["Importance"])
st.pyplot(fig)
