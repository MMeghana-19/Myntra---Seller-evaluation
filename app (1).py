import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load the pickled components
def load_pickle_files():
    with open('random_forest_modelC.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scalerC.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('ordinal_encoderC.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)

    with open('label_encodersC.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('class_mappingC.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

    return model, scaler, ordinal_encoder, label_encoders, class_mapping

# Load all the components
model, scaler, ordinal_encoder, label_encoders, class_mapping = load_pickle_files()

# Function to preprocess user input
def preprocess_input(user_input):
    # Encode categorical features using label encoders
    user_input['gender'] = label_encoders['gender'].get_loc(user_input['gender'])
    user_input['category'] = label_encoders['category'].get_loc(user_input['category'])
    user_input['sub_category'] = label_encoders['sub_category'].get_loc(user_input['sub_category'])
    user_input['name'] = label_encoders['name'].get_loc(user_input['name'])
    user_input['seller'] = label_encoders['seller'].get_loc(user_input['seller'])

    # Apply ordinal encoding for price_category, discount_category, brand_performance
    user_input[['price_category', 'discount_category', 'brand_performance']] = ordinal_encoder.transform(user_input[['price_category', 'discount_category', 'brand_performance']])

    # Scaling numeric features
    numeric_columns = ['price', 'mrp', 'ratingTotal', 'product_avg_rating']
    user_input[numeric_columns] = scaler.transform(user_input[numeric_columns])

    return user_input

# Streamlit UI for input
st.title('Seller Performance Evaluation & Prediction')
st.write("Enter the seller & product details below to predict its performance:")

# Input fields for the user
gender = st.selectbox('Gender', ['Male', 'Female', 'Unisex'])
category = st.selectbox('Category', ['Care & Beauty', 'Clothing', 'Footwear & Bag', 'Accessories', 'Home & Living', 'Miscellaneous'])
name = st.text_input('Product Name')
seller = st.text_input('Brand Name')
product_avg_rating = st.number_input('Product Average Rating', min_value=0.0, max_value=5.0, step=0.1)
price_category = st.selectbox('Price Category', ['Low', 'Medium', 'High'])
discount_category = st.selectbox('Discount Category', ['None', 'Low', 'Medium', 'High'])

# When the user clicks on 'Predict' button
if st.button('Evaluate/Predict'):
    # Prepare user input as DataFrame
    user_input = pd.DataFrame({
        'gender': [gender.lower()],
        'category': [category.lower()],
        'name': [name.lower()],
        'seller': [seller.lower()],
        'product_avg_rating': [product_avg_rating.lower()],
        'price_category': [price_category.lower()],
        'discount_category': [discount_category.lower()]
    })

    # Preprocess input data
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(processed_input)

    # Convert numeric prediction to human-readable label
    predicted_label = class_mapping[prediction[0]]

    # Display the result
    st.write(f"Predicted Brand Performance: {predicted_label}")
