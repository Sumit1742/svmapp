import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load model and scaler (make sure both are saved)
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save scaler if you used it

# Get feature names
data = load_breast_cancer()
feature_names = data.feature_names
class_names = data.target_names

st.title('Breast Cancer Prediction (SVM Model)')

# Dynamic form for user input
user_input = []
for feature in feature_names:
    val = st.number_input(f'Enter value for {feature}', value=0.0)
    user_input.append(val)

# Prediction button
if st.button('Predict'):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)  # Use same scaling as training
    prediction = svm.predict(X_scaled)[0]
    st.write(f'Predicted class: {class_names[prediction]}')
