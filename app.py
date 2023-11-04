# app.py
import streamlit as st
import pandas as pd
from sklearn import preprocessing
import joblib
import numpy as np


model = joblib.load("lr_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")





# Create a title for the app.
st.title('Used Car Price Prediction')

# Create input fields for the user to enter the features of the car.
manufacturer = st.text_input('Manufacturer')
year = st.number_input('Year')
condition = st.selectbox('Condition', options=['Nigerian Used', 'Foriegn Used', 'Brand New'])
mileage = st.number_input('Mileage (km)')
engine_size = st.number_input('Engine size (L)')
fuel = st.selectbox('Fuel type', options=['Petrol', 'Diesel', 'Hybrid', 'Electric'])
transmission = st.selectbox('Transmission type', options=['Automatic', 'Manual', 'CVT', 'AMT'])

# Create a button to trigger the prediction function.
if st.button('Predict price'):
    # Make a prediction using the model.
   

# Assuming you have your data in separate variables like manufacturer, year, condition, mileage, engine_size, fuel, transmission
    data = pd.DataFrame({'Make': [manufacturer], 'Year of manufacture': [year], 'Condition': [condition], 'Mileage': [mileage],
                        'Engine Size': [engine_size], 'Fuel': [fuel], 'Transmission': [transmission]})

    # Assuming 'preprocessor' is a pre-trained preprocessing pipeline, and 'model' is a pre-trained machine learning model
    transformed_data = preprocessor.transform(data)
    prediction = model.predict(transformed_data)


    # Display the predicted price to the user.
    st.write('Predicted price: ₦{:.2f}'.format(prediction[0]))



