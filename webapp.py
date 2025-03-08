import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained models
models = {
    "Linear Regression": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\Linear_Regression.joblib"),
    "KNN": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\KNN.joblib"),
    "Decision Tree": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\Decision_Tree.joblib"),
    "Random Forest": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\Random_Forest.joblib"),
    "Support Vector Regression": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\Support_Vector_Regression.joblib"),
    "Gradient Boosting": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\Gradient_Boosting.joblib"),
    "XGBoost": joblib.load("C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\XGBoost.joblib")
}

# Set up the Streamlit interface
st.title("Solar Power Prediction App")
st.markdown("Enter the following parameters to predict solar power output.")

# Input fields for user data
daily_yield = st.number_input("Daily Yield (kWh)", min_value=0.0, value=1000.0)
total_yield = st.number_input("Total Yield (kWh)", min_value=0.0, value=500000.0)
ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
irradiation = st.number_input("Solar Irradiation (W/m²)", min_value=0.0, max_value=1000.0, value=800.0)

# Dropdown to select the model
model_name = st.selectbox("Select Model", list(models.keys()))

# Prediction button
if st.button("Predict Solar Power Output"):
    # Prepare input data for prediction
    input_data = np.array([[daily_yield, total_yield, ambient_temp, irradiation]])
    
    # Make prediction using the selected model
    model = models[model_name]
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"Predicted Solar Power Output: {prediction[0]:.2f} kW")
    
    # Optional: Display a simple bar chart of the input parameters
    st.subheader("Input Parameters")
    params = ['Daily Yield', 'Total Yield', 'Ambient Temperature', 'Irradiation']
    values = [daily_yield, total_yield, ambient_temp, irradiation]
    
    plt.bar(params, values, color='skyblue')
    plt.ylabel('Value')
    plt.title('Input Parameters for Prediction')
    st.pyplot(plt)

# Clear button to reset inputs
if st.button("Clear Inputs"):
    st.experimental_rerun()

# Run the app
# Use the command: streamlit run "C:\ALL\codes\Python\edunet\project\Solar-Power-Generation-Forecasting-main\webapp.py"

