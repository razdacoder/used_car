# app.py
import streamlit as st
import pandas as pd
from sklearn import preprocessing


# Load your model
import joblib  # Replace with the appropriate library for your model

st.title("Used Car Price Prediction")


# Load your trained model
preprocessor_model = joblib.load("preprocessor.pkl")  # Replace with your model file
model = joblib.load("RidgeCV_Model.pkl")

# Input fields for the features
location = st.selectbox("Location", ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur',
       'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'])  # Replace with actual locations
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])  # Replace with actual fuel types
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])  # Replace with actual transmission types
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])  # Replace with actual owner types
brand = st.text_input("Brand")
year = st.number_input("Year")
kilometers_driven = st.number_input("Kilometers Driven")
mileage = st.number_input("Mileage")
engine = st.number_input("Engine")
power = st.number_input("Power")
seats = st.number_input("Seats")

if st.button("Predict Price"):
    # Create a dictionary from the user input
    
    user_data = {
        'Location': location,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Brand': brand,
        'Year': year,
        'Kilometers_Driven': kilometers_driven,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Seats': seats
    }

    # Create a DataFrame from the user data
    input_data = pd.DataFrame([user_data])
    preprocessed_data = preprocessor_model.transform(input_data)
    print("yeh")
    prediction = model.predict(preprocessed_data)
    print("yehh")
    st.write(f"Predicted Price: ₦{abs(prediction[0].round(2) * 9.20).round(2)}")
