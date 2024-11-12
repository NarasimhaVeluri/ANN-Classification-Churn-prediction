## Load the required libraries

import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

## Load the trained model

model = load_model('model.h5')

## Load the encoder and scaler pickle file

with open("label_encode_gender.pkl",'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("encoder.pkl",'rb') as file:
    encoder = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


## Streamlit App

st.title("Customer Churn Prediction")

import streamlit as st
import pandas as pd

# Set the title of the app
st.title("Bank Customer Data Input")

# Define the input fields for each column
credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography",encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age",18,92)
tenure = st.number_input("Tenure",0,10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_credit_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary")


# Create a dictionary of the input values
input_data = {
    "CreditScore":[credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}
## One hot encodong geography
geo_encoded = encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder.get_feature_names_out(['Geography']))


input_data_df = pd.DataFrame(input_data)
input_data_df = pd.concat([input_data_df.reset_index(drop=True), geo_encoded_df],axis = 1)

## Scale the input data

scaled_df = scaler.transform(input_data_df)
scaled_df

## Predict the churn

prediction = model.predict(scaled_df)

prediction_proba = prediction[0][0]

st.write(f"churn_probability:{prediction_proba:.2f}")
if prediction_proba>0.5:
    st.write("The customer is likely to churn")

else:
    st.write("The customer is not likely to churn")