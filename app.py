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

/* Input box styling */
div[data-baseweb="input"] {
    border-radius: 10px;
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
    background: linear-gradient(90deg, #16a34a, #15803d);
}

/* Prediction box */
.success {
    background: rgba(34, 197, 94, 0.2);
    padding: 15px;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

</style>
""", unsafe_allow_html=True)



import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATASET
# =========================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# UI
# =========================
st.title("🌲 Random Forest Classification")

st.write("Enter all feature values:")

inputs = []

# 🔥 IMPORTANT: USE ALL FEATURES (30)
for feature in data.feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# =========================
# PREDICTION
# =========================
if st.button("Predict 🔍"):

    arr = np.array(inputs).reshape(1, -1)  # NOW 30 features

    arr = scaler.transform(arr)

    prediction = model.predict(arr)

    if prediction[0] == 0:
        st.success("🟢 Benign (No Cancer)")
    else:
        st.error("🔴 Malignant (Cancer)")
