import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib

# Import all preprocessing functions
from preprocessing import (
    change_data_type,
    impute_missing_values,
    filter_outliers,
    feature_engineering,
    feature_encoding,
    drop_columns,
    feature_scaling
)

# Page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation Analysis")

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('kmeans_model.joblib')

model = load_model()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file, index_col=0)
    
    # Convert Dt_Customer to datetime immediately after loading
    try:
        data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    except Exception as e:
        st.error(f"Error converting Dt_Customer to datetime: {str(e)}")
        st.stop()
    
    st.subheader("Raw Data Preview")
    st.write(data.head())
    
    # Display initial missing values
    st.subheader("Missing Values Summary")
    missing_values = data.isnull().sum()
    if missing_values.any():
        st.write(missing_values[missing_values > 0])
    else:
        st.write("No missing values found in the dataset.")

    # Preprocessing steps in sidebar
    st.sidebar.header("Preprocessing Steps")
    
    # 1. Data Type Conversion
    st.sidebar.subheader("1. Data Type Conversion")
    date_columns = st.sidebar.multiselect(
        "Select date columns",
        data.columns,
        default=[col for col in data.columns if 'date' in col.lower() or 'dt' in col.lower()]
    )
    date_format = st.sidebar.text_input("Date format (e.g., %d-%m-%Y)", "%d-%m-%Y")
    
    # 2. Missing Values Imputation
    st.sidebar.subheader("2. Missing Values Imputation")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_impute = st.sidebar.multiselect(
        "Select columns for missing value imputation",
        numeric_columns,
        default=numeric_columns[data[numeric_columns].isnull().any()].tolist()
    )
    
    # 3. Outlier Handling
    st.sidebar.subheader("3. Outlier Handling")
    outlier_columns = st.sidebar.multiselect(
        "Select columns for outlier removal",
        numeric_columns
    )
    
    # 4. Feature Engineering options
    st.sidebar.subheader("4. Feature Engineering")
    st.sidebar.write("Features to be engineered:")
    st.sidebar.markdown("""
    - Age (from Year_Birth)
    - Age_Group (Young/Middle/Senior Adult)
    - Membership_Duration
    - Total_Acc_Camp
    - Total_Spending
    - Total_Purchases
    - CVR (Conversion Rate)
    """)
    do_feature_engineering = st.sidebar.checkbox("Apply Feature Engineering", value=True)
    
    # Custom age group thresholds
    if do_feature_engineering:
        st.sidebar.write("Age Group Thresholds:")
        middle_age_threshold = st.sidebar.slider("Middle Adult Age Threshold", 30, 50, 40)
        senior_age_threshold = st.sidebar.slider("Senior Adult Age Threshold", 50, 80, 60)
    
    # Create a temporary dataframe for feature engineering preview
    temp_data = data.copy()
    
    # Ensure Dt_Customer is datetime type in temp_data
    if 'Dt_Customer' in temp_data.columns and not pd.api.types.is_datetime64_any_dtype(temp_data['Dt_Customer']):
        temp_data['Dt_Customer'] = pd.to_datetime(temp_data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    
    if do_feature_engineering:
        try:
            feature_engineering(temp_data)
            # Update Age_Group based on custom thresholds
            temp_data['Age'] = datetime.now().year - temp_data['Year_Birth']
            def custom_age_group(x):
                if x >= senior_age_threshold:
                    return 'Senior Adult'
                elif x >= middle_age_threshold:
                    return 'Middle Adult'
                else:
                    return 'Young Adult'
            temp_data['Age_Group'] = temp_data['Age'].apply(custom_age_group)
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
            st.stop()
    
    # 5. Feature Encoding
    st.sidebar.subheader("5. Feature Encoding")
    # Get both original and engineered categorical columns
    original_categorical = data.select_dtypes(include=['object']).columns.tolist()
    all_categorical = temp_data.select_dtypes(include=['object']).columns.tolist()
    
    # Show all available categorical columns for encoding
    columns_to_encode = st.sidebar.multiselect(
        "Select categorical columns to encode",
        all_categorical,
        default=original_categorical
    )
    
    # 6. Column Dropping
    st.sidebar.subheader("6. Drop Columns")
    all_columns = temp_data.columns.tolist()
    columns_to_drop = st.sidebar.multiselect(
        "Select columns to drop",
        all_columns
    )
    
    # Process button
    if st.sidebar.button("Apply Preprocessing"):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Copy of original data
            processed_data = data.copy()
            
            # 1. Date conversion
            status_text.text("1. Converting date columns...")
            for col in date_columns:
                processed_data = change_data_type(processed_data, col, 'datetime', date_format)
            progress_bar.progress(14)
            
            # 2. Impute missing values
            status_text.text("2. Imputing missing values...")
            if columns_to_impute:
                processed_data = impute_missing_values(processed_data, columns_to_impute)
            progress_bar.progress(28)
            
            # 3. Handle outliers
            status_text.text("3. Handling outliers...")
            if outlier_columns:
                processed_data = filter_outliers(processed_data, outlier_columns)
            progress_bar.progress(42)
            
            # 4. Feature engineering
            status_text.text("4. Performing feature engineering...")
            if do_feature_engineering:
                # Ensure Dt_Customer is datetime
                if not pd.api.types.is_datetime64_any_dtype(processed_data['Dt_Customer']):
                    processed_data['Dt_Customer'] = pd.to_datetime(processed_data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
                
                feature_engineering(processed_data)
                # Update Age_Group based on custom thresholds
                processed_data['Age'] = datetime.now().year - processed_data['Year_Birth']
                processed_data['Age_Group'] = processed_data['Age'].apply(custom_age_group)
                
                # Display feature engineering results
                st.subheader("Feature Engineering Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Age Group Distribution")
                    st.write(processed_data['Age_Group'].value_counts())
                
                with col2:
                    st.write("Total Spending Statistics")
                    st.write(processed_data['Total_Spending'].describe())
                
                with col3:
                    st.write("Conversion Rate Statistics")
                    st.write(processed_data['CVR'].describe())
            
            progress_bar.progress(56)
            
            # 5. Categorical encoding
            status_text.text("5. Encoding categorical features...")
            if columns_to_encode:
                processed_data = feature_encoding(processed_data, columns_to_encode)
            progress_bar.progress(70)
            
            # 6. Drop columns
            status_text.text("6. Dropping selected columns...")
            if columns_to_drop:
                processed_data = drop_columns(processed_data, columns_to_drop)
            progress_bar.progress(84)
            
            # 7. Feature scaling
            status_text.text("7. Scaling features...")
            processed_data = feature_scaling(processed_data)
            progress_bar.progress(95)
            
            # Make predictions
            status_text.text("Making predictions...")
            clusters = model.predict(processed_data)
            processed_data['Cluster'] = clusters
            progress_bar.progress(100)
            status_text.text("Preprocessing complete!")
            
            # Display results
            st.subheader("Processed Data Preview")
            st.write(processed_data.head())
            
            # Show shape of data before and after preprocessing
            st.subheader("Data Shape")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Shape:", data.shape)
            with col2:
                st.write("Processed Shape:", processed_data.shape)
            
            # Cluster Analysis
            st.subheader("Cluster Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster distribution
                st.write("Cluster Distribution")
                fig, ax = plt.subplots()
                processed_data['Cluster'].value_counts().plot(kind='bar')
                plt.title("Distribution of Clusters")
                plt.xlabel("Cluster")
                plt.ylabel("Count")
                st.pyplot(fig)
            
            with col2:
                # PCA visualization
                st.write("PCA Visualization")
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(processed_data.drop('Cluster', axis=1))
                
                fig, ax = plt.subplots()
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=processed_data['Cluster'], cmap='viridis')
                plt.colorbar(scatter)
                plt.title("Cluster Visualization (PCA)")
                plt.xlabel("First Principal Component")
                plt.ylabel("Second Principal Component")
                st.pyplot(fig)
            
            # Cluster Profiles with Age Group Analysis
            st.subheader("Cluster Profiles")
            for cluster in processed_data['Cluster'].unique():
                with st.expander(f"Cluster {cluster} Profile"):
                    cluster_data = processed_data[processed_data['Cluster'] == cluster]
                    
                    # General statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Size: {len(cluster_data)} customers")
                        st.write("Key Statistics:")
                        st.write(cluster_data.describe())
                    
                    with col2:
                        if 'Age_Group' in cluster_data.columns:
                            st.write("Age Group Distribution:")
                            age_dist = cluster_data['Age_Group'].value_counts()
                            fig, ax = plt.subplots()
                            age_dist.plot(kind='pie', autopct='%1.1f%%')
                            plt.title(f"Age Groups in Cluster {cluster}")
                            st.pyplot(fig)
                            
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Error occurred during preprocessing")

# Add information about the app
with st.expander("About this app"):
    st.write("""
    This app performs customer segmentation using K-means clustering. The preprocessing steps are:
    1. Data type conversion
    2. Missing values imputation
    3. Outlier handling
    4. Feature engineering (including Age Group categorization)
    5. Feature encoding
    6. Column dropping
    7. Feature scaling
    
    Upload your CSV file and follow the preprocessing steps in the sidebar to begin.
    """)