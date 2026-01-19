import streamlit as st
import joblib
import pandas as pd

model = joblib.load('Heart_Disease_Model.pkl')

st.title("Heart Disease Predictor")
st.info("AUC 0.921 | Beats ECG 60-70% sensitivity")

thal = st.selectbox("Thallium Stress", [3, 6, 7])
exang = st.selectbox("Exercise Angina", [0, 1])
ca = st.slider("Blocked Vessels", 0, 3, 0)

if st.button("🔬 Predict Risk"):

    #Needs the same attributes from the one that the model was built on.
    data = pd.DataFrame({'Thallium': [thal], 'Exercise_angina': [exang], 'Number_of_vessels_fluro': [ca]})
    prob = model.predict(data)[0] #statsmodels doesnt have predict-proba
    st.metric("Disease Risk", f"{prob:.1%}")
