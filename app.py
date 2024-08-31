import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset from a GitHub URL or from the local file if using locally
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gauthamgtg/Used-Car-Price-Predictor/main/PakWheel%20used%20Car%20Data.csv"
    car_data = pd.read_csv(url)
    car_data = car_data.drop('description', axis=1)
    car_data = car_data.drop('url', axis=1)

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

car_data = load_data()

# Prepare features and target variable
X = car_data.drop('price', axis=1)
y = car_data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

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

# Model performance
st.write(f"Model Mean Squared Error: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
