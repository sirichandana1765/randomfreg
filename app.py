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
