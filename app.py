import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0b1220, #111827);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

/* Smooth animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title styling */
h1, h2, h3 {
    color: #38bdf8;
    text-align: center;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# RANDOM FOREST REGRESSOR
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# MODEL SCORE
# =========================
y_pred = model.predict(X_test_scaled)

score = r2_score(y_test, y_pred)

# =========================
# UI
# =========================
st.title("🌲 Random Forest Regression")

st.write("Enter feature values:")

inputs = []

for feature in data.feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# =========================
# PREDICTION
# =========================
if st.button("Predict 🔍"):

    arr = np.array(inputs).reshape(1, -1)

    arr_scaled = scaler.transform(arr)

    prediction = model.predict(arr_scaled)

    st.success(f"Predicted House Price: ${prediction[0] * 100000:.2f}")

# =========================
# SHOW MODEL PERFORMANCE
# =========================
st.write(f"### Model R² Score: {score:.2f}")
