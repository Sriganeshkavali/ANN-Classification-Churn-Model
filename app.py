import streamlit as st
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle



# Load the model
model= tf.keras.models.load_model('churn_model.h5')

# Load the scaler and label encoder
with open('Sclaled_file.pkl', 'rb') as f:
    scaler = pickle.load(f)


with open('label_enoded_file.pkl', 'rb') as f:
    label_enc= pickle.load(f)

# Load the one-hot encoder
with open('one_hot_emcoding_file.pkl', 'rb') as f:
    one_hot_enc= pickle.load(f)

## streamlit app

st.title("Churn Prediction App")

st.write("Enter the customer details to predict churn probability")
# Input fields for customer details
geography=st.selectbox("Geography",one_hot_enc.categories_[0])
gender=st.selectbox('gender',label_enc.classes_)
age=st.slider('Age',18,100,30)
balance=st.number_input('Credit Score')
credit_score=st.number_input('Balance')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])


# prepare the input data
# example input data
input=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_enc.transform([gender])[0]],
    'Age':[age], 
    'Tenure':[tenure], 
    'Balance':[balance], 
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card], 
    'IsActiveMember':[is_active_member], 
    'EstimatedSalary':[estimated_salary]
})

# one hot encoding for geography
geography_encoded = one_hot_enc.transform([[geography]])
geo_df= pd.DataFrame(geography_encoded, columns=one_hot_enc.get_feature_names_out(['Geography']))


# concatenate the geography encoded data with the input data
input_data = pd.concat([input, geo_df], axis=1)

#scaling the input data
input_data_scaled = scaler.transform(input_data)



# prediction churn probability
prediction = model.predict(input_data_scaled)

prediction_prob= prediction[0][0]



if prediction_prob > 0.5:
    st.write("Customer is likely to churn ")
else:
    st.write("Customer is not likely to churn")


