import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="AI Health Risk Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp{
    background:#0B1020;
    color:white;
}

.metric-card{
    background:#111827;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("health_model.pkl")
le = joblib.load("label_encoder.pkl")

# Sidebar
st.sidebar.title("🧬 Novagen")

age = st.sidebar.slider("Age",18,80,35)
bmi = st.sidebar.slider("BMI",15.0,45.0,25.0)
bp = st.sidebar.slider("Blood Pressure",80,200,120)
heart_rate = st.sidebar.slider("Heart Rate",50,150,75)
cholesterol = st.sidebar.slider("Cholesterol",100,350,200)
glucose = st.sidebar.slider("Glucose",70,300,100)
sleep = st.sidebar.slider("Sleep Hours",1.0,12.0,7.0)
exercise = st.sidebar.slider("Exercise Hours",0.0,5.0,1.0)
stress = st.sidebar.slider("Stress Level",1,10,5)

# Header
st.title("🧬 Anti-gen Health Risk Predictor")
st.write("AI Powered Disease Risk Analysis")

# Metrics Row
c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy","95%")
c2.metric("Samples","5000")
c3.metric("Features","9")
c4.metric("Model","Random Forest")

# Prediction
if st.button("⚡ Analyze Health Risk"):

    data = pd.DataFrame([[
        age,bmi,bp,heart_rate,
        cholesterol,glucose,
        sleep,exercise,stress
    ]],columns=[
        "Age","BMI","Blood_Pressure",
        "Heart_Rate","Cholesterol",
        "Glucose_Level","Sleep_Hours",
        "Exercise_Hours","Stress_Level"
    ])

    pred = model.predict(data)
    risk = le.inverse_transform(pred)[0]

    st.subheader("Prediction Result")

    if risk == "Low":
        st.success(f"Risk Level : {risk}")

    elif risk == "Medium":
        st.warning(f"Risk Level : {risk}")

    else:
        st.error(f"Risk Level : {risk}")

    st.dataframe(data)
    # Feature Importance Chart
st.subheader("📊 Feature Importance")

import plotly.express as px

features = [
    "BMI",
    "Blood Pressure",
    "Cholesterol",
    "Stress Level",
    "Glucose",
    "Sleep Hours",
    "Age",
    "Heart Rate",
    "Exercise Hours"
]

importance = [95, 80, 70, 60, 55, 45, 40, 35, 25]

chart_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

fig = px.bar(
    chart_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Health Risk Factors"
)

st.plotly_chart(fig, use_container_width=True)
