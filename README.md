# Solar Power Prediction App

This project is a web application built using Streamlit that predicts solar power output based on various input parameters. The application utilizes multiple machine learning models to provide accurate predictions, allowing users to select the model they prefer.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Models Used](#models-used)
- [Contributing](#contributing)
- [License](#license)

## Features

- User-friendly interface for inputting parameters.
- Multiple machine learning models for prediction.
- Visualization of input parameters.
- Clear button to reset inputs easily.
- Real-time predictions of solar power output.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib




Enter the required input parameters and select the desired model from the dropdown menu.

Click the "Predict Solar Power Output" button to see the predicted output.

Use the "Clear Inputs" button to reset the input fields.

Input Parameters
The application requires the following input parameters:

Daily Yield (kWh): The amount of solar energy produced in a day.
Total Yield (kWh): The total amount of solar energy produced over the lifetime of the solar installation.
Ambient Temperature (°C): The temperature of the environment where the solar panels are located.
Solar Irradiation (W/m²): The solar radiation received by the solar panels.
Models Used
The application utilizes the following machine learning models:

Linear Regression,
K-Nearest Neighbors (KNN),
Decision Tree Regressor,
Support Vector Regression (SVR),
Gradient Boosting Regressor,
XGBoost Regressor,
Random forest Regressor( this model is in archive fomat in two zip files exact them to get this model)

