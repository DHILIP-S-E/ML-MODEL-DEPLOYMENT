import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI for car features
st.title("Car Price Prediction")

year = st.slider("Year of Manufacture", 2010, 2022, 2015)
km_driven = st.slider("Kilometers Driven", 0, 200000, 50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])

# Convert Fuel Type to numeric (0: Petrol, 1: Diesel)
fuel_type = 0 if fuel_type == "Petrol" else 1

# Predict price when the button is clicked
if st.button("Predict"):
    input_data = np.array([[year, km_driven, fuel_type]])
    predicted_price = model.predict(input_data)
    st.success(f"Predicted Car Price: ₹{predicted_price[0]:,.2f}")
