# """
# Steamlit app for making clustering of input data
# """

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib # or `import pickle` if using pickle

# Load pre-trained model
model = joblib.load('kmeans_model.joblib')  # or `pickle.load(open('kmeans_model.pkl', 'rb'))`

st.title("K-Means Clustering with Pre-trained Model")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())
    
    # Preprocessing
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=[float, int]))
    
    # Predict clusters with the loaded model
    clusters = model.predict(data_scaled)
    data['Cluster'] = clusters
    
    # Display clustered data
    st.write("Clustered Data", data.head())
