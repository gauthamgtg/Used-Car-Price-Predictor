import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset (with caching to speed up)
@st.cache_data
def load_data():
    car_data = pd.read_csv('PakWheel used Car Data.csv')
    
    # Data Cleaning
    car_data['engineDisplacement'] = car_data['engineDisplacement'].str.replace('cc', '').str.strip()
    car_data['engineDisplacement'] = pd.to_numeric(car_data['engineDisplacement'], errors='coerce')
    car_data['mileageFromOdometer'] = car_data['mileageFromOdometer'].str.replace(' km', '').str.replace(',', '').str.strip()
    car_data['mileageFromOdometer'] = pd.to_numeric(car_data['mileageFromOdometer'], errors='coerce')
    
    # Drop rows with missing values
    car_data.dropna(inplace=True)
    
    # Convert categorical columns to dummy variables
    categorical_columns = ['Model', 'City', 'manufacturer', 'vehicleTransmission', 'fuelType', 'itemCondition']
    car_data = pd.get_dummies(car_data, columns=categorical_columns, drop_first=True)
    
    return car_data

# Load the pre-trained model (cached for speed)
@st.cache_resource
def load_model():
    return joblib.load('car_price_predictor_model.pkl')

# Load data and model
car_data = load_data()
model = load_model()

# Prepare input features
X = car_data.drop('price', axis=1)

# Streamlit App
st.title("Used Car Price Prediction")

# User Inputs
st.header("Enter Car Details:")
engine_displacement = st.number_input("Engine Displacement (cc):", min_value=500, max_value=8000, step=100)
mileage = st.number_input("Mileage from Odometer (km):", min_value=0, max_value=500000, step=1000)

# User selections for categorical variables
city = st.selectbox("City:", sorted([col.replace('City_', '') for col in X.columns if 'City_' in col]))
model_selected = st.selectbox("Model:", sorted([col.replace('Model_', '') for col in X.columns if 'Model_' in col]))
fuel_type = st.selectbox("Fuel Type:", sorted([col.replace('fuelType_', '') for col in X.columns if 'fuelType_' in col]))

# Prepare the input data for prediction
input_data = pd.DataFrame([[engine_displacement, mileage]], columns=['engineDisplacement', 'mileageFromOdometer'])

# Add dummy variables for categorical inputs
for col in X.columns:
    input_data[col] = 0  # Initialize all columns to 0
    
# Set the corresponding columns to 1 based on user input
if f"City_{city}" in input_data.columns:
    input_data[f"City_{city}"] = 1
if f"Model_{model_selected}" in input_data.columns:
    input_data[f"Model_{model_selected}"] = 1
if f"fuelType_{fuel_type}" in input_data.columns:
    input_data[f"fuelType_{fuel_type}"] = 1

# Predict the price
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.subheader(f"Estimated Price: {predicted_price:,.2f}")

