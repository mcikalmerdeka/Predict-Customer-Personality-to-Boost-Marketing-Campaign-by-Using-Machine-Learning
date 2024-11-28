import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all preprocessing functions
from preprocessing import (
    check_data_information,
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

# Input type selection
input_type = st.radio('Select Input Type', ['File Upload', 'Manual Input'])

# Initialize data variable
data = None

if input_type == 'File Upload':
    # File upload
    uploaded_data = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_data:
        try:
            # Load data from file upload
            file_data = pd.read_csv(uploaded_data, index_col=0)
            
            st.subheader("Raw Data Preview")
            st.write(file_data.head())
            
            # Display data information
            st.subheader("Data Information")
            st.write(check_data_information(file_data, file_data.columns))
            
            # Create a copy for preprocessing
            data = file_data.copy()

        except Exception as e:
            st.error(f"Error loading the file: {e}")

else:  # Manual Input
    # Provide a sample file upload for reference columns
    st.subheader("First, upload a reference CSV file")
    reference_file = st.file_uploader("Choose a reference CSV file", type="csv", key="reference_upload")
    
    if reference_file:
        try:
            # Load reference data
            reference_data = pd.read_csv(reference_file, index_col=0)
            
            st.subheader("Enter Customer Data")
            
            # Create a dictionary to store input values
            input_data = {}
            
            # Dynamically create input fields based on column types
            for column in reference_data.columns:
                # Handle different column types
                if pd.api.types.is_datetime64_any_dtype(reference_data[column]):
                    # Date input for datetime columns
                    input_data[column] = st.date_input(f'Enter {column}')
                
                elif pd.api.types.is_numeric_dtype(reference_data[column]):
                    # Number input for numeric columns
                    col_min = reference_data[column].min()
                    col_max = reference_data[column].max()
                    col_mean = reference_data[column].mean()
                    
                    input_data[column] = st.number_input(
                        f'Enter {column}', 
                        min_value=float(col_min), 
                        max_value=float(col_max), 
                        value=float(col_mean),
                        step=0.1
                    )
                
                elif pd.api.types.is_categorical_dtype(reference_data[column]) or reference_data[column].dtype == 'object':
                    # Selectbox for categorical/object columns
                    unique_values = reference_data[column].unique()
                    input_data[column] = st.selectbox(
                        f'Select {column}', 
                        options=list(unique_values)
                    )
                
                else:
                    # Fallback for any other types
                    input_data[column] = st.text_input(f'Enter {column}')
            
            # Submit button to process the input
            if st.button('Submit Data'):
                # Convert input to DataFrame
                manual_data = pd.DataFrame([input_data])
                
                st.subheader("Entered Data")
                st.dataframe(manual_data)
                
                # Create a copy for preprocessing
                data = manual_data.copy()

        except Exception as e:
            st.error(f"Error loading the reference file: {e}")
    else:
        st.info("Please upload a reference CSV file to see input fields")

# Preprocessing steps (only proceed if data is not None)
if data is not None:

    # Preprocessing steps in sidebar
    st.sidebar.header("Preprocessing Steps")
    
    # 1. Data Type Conversion
    st.sidebar.subheader("1. Data Type Conversion")
    date_columns = st.sidebar.multiselect(
        "Select date columns",
        options=data.columns,
        default=[col for col in data.columns if 'date' in col.lower() or 'dt' in col.lower()]
    )
    date_format = st.sidebar.text_input("Date format (e.g., %d-%m-%Y)", "%d-%m-%Y")
    
    # 2. Missing Values Imputation
    st.sidebar.subheader("2. Missing Values Imputation")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_impute = st.sidebar.multiselect(
        "Select columns for missing value imputation",
        options=numeric_columns,
        default=numeric_columns[data[numeric_columns].isnull().any()].tolist()
    )
    
    # 3. Outlier Handling
    st.sidebar.subheader("3. Outlier Handling")
    outlier_columns = st.sidebar.multiselect(
        "Select columns for outlier removal",
        options=numeric_columns,
        default=['Year_Birth', 'Income', 'MntMeatProducts', 'MntSweetProducts', 'NumWebPurchases', 'NumCatalogPurchases']
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
    
    # 5. Feature Encoding
    st.sidebar.subheader("5. Feature Encoding")

    if data is not None:  # Only show encoding options if data exists
        try:
            # Create a temporary dataframe for feature engineering preview
            temp_data = data.copy()
            
            # Ensure Dt_Customer is datetime type in temp_data before feature engineering
            if 'Dt_Customer' in temp_data.columns and not pd.api.types.is_datetime64_any_dtype(temp_data['Dt_Customer']):
                temp_data['Dt_Customer'] = pd.to_datetime(temp_data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
            
            # Apply feature engineering if enabled
            if do_feature_engineering:
                feature_engineering(temp_data, middle_age_threshold, senior_age_threshold)
            
            # Get all categorical columns including engineered ones
            all_categorical = temp_data.select_dtypes(include=['object']).columns.tolist()
            
            # Get original categorical columns that exist in all_categorical
            original_categorical = [col for col in data.select_dtypes(include=['object']).columns 
                                  if col in all_categorical]
            
            # Show all available categorical columns for encoding
            columns_to_encode = st.sidebar.multiselect(
                "Select categorical columns to encode",
                options=all_categorical,
                default=all_categorical,
                key="encoding_selector"
            )
            
        except Exception as e:
            st.sidebar.error(f"Error in feature encoding setup: {str(e)}")
            columns_to_encode = []
            
    else:
        st.sidebar.info("Please upload or input data to see encoding options")
        columns_to_encode = []

    # 6. Column Dropping
    st.sidebar.subheader("6. Drop Columns")
    all_columns = temp_data.columns.tolist()
    columns_to_drop = st.sidebar.multiselect(
        "Select columns to drop",
        options=all_columns,
        default=['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Response']
    )
    
    # Process button
    if st.sidebar.button("Apply Preprocessing"):
        # Create a container for progress tracking
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### Preprocessing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Keep copy of data
                processed_data = data.copy()

                # 1. Date conversion
                status_text.markdown("**Step 1/7:** Converting date columns...")
                for col in date_columns:
                    processed_data = change_data_type(processed_data, col, 'datetime', date_format)
                progress_bar.progress(14)
                st.success("Date conversion completed successfully!")
                st.write("After Date Conversion:")
                st.write(processed_data.head(3))
                
                # 2. Impute missing values
                status_text.markdown("**Step 2/7:** Imputing missing values...")
                if columns_to_impute:
                    processed_data[columns_to_impute] = impute_missing_values(processed_data, columns_to_impute)
                progress_bar.progress(28)
                st.success("Missing value imputation completed successfully!")
                st.write("After Missing Value Imputation:")
                st.write(processed_data.head(3))
                
                # 3. Handle outliers
                status_text.markdown("**Step 3/7:** Handling outliers...")
                if outlier_columns:
                    processed_data = filter_outliers(processed_data, outlier_columns)
                progress_bar.progress(42)
                st.success("Outlier handling completed successfully!")
                st.write("After Outlier Handling:")
                st.write(processed_data.head(3))
                
                # 4. Feature engineering
                status_text.markdown("**Step 4/7:** Performing feature engineering...")
                if do_feature_engineering:
                    feature_engineering(processed_data, middle_age_threshold, senior_age_threshold)
                progress_bar.progress(56)
                st.success("Feature engineering completed successfully!")
                st.write("After Feature Engineering:")
                st.write(processed_data.head(3))
                
                # 5. Categorical encoding
                status_text.markdown("**Step 5/7:** Encoding categorical features...")
                if columns_to_encode:
                    # Use processed_data for encoding
                    st.write("Columns to encode:", columns_to_encode)
                    # st.write("Available columns before encoding:", processed_data.columns.tolist())
                    encoded_data = feature_encoding(processed_data, columns_to_encode)
                    processed_data = encoded_data.copy()
                progress_bar.progress(70)
                st.success("Categorical encoding completed successfully!")
                st.write("After Categorical Encoding:")
                st.write(processed_data.head(3))
                
                # 6. Drop columns
                status_text.markdown("**Step 6/7:** Dropping selected columns...")
                if columns_to_drop:
                    processed_data = drop_columns(processed_data, columns_to_drop)
                progress_bar.progress(84)
                st.success("Column dropping completed successfully!")
                st.write("After Dropping Columns:")
                st.write(processed_data.head(3))
                
                # 7. Feature scaling
                status_text.markdown("**Step 7/7:** Scaling features...")
                processed_data = feature_scaling(processed_data)
                progress_bar.progress(100)
                st.success("Feature scaling completed successfully!")
                st.write("After Feature Scaling:")
                st.write(processed_data.head(3))

                # Add completion message and separator
                st.markdown("---")  # Horizontal line separator
                st.success("🎉 All preprocessing steps completed successfully! The data is now ready for clustering.")
                st.markdown("""
                ### Preprocessing Summary:
                1. ✅ Date Conversion
                2. ✅ Missing Value Imputation
                3. ✅ Outlier Handling
                4. ✅ Feature Engineering
                5. ✅ Categorical Encoding
                6. ✅ Column Dropping
                7. ✅ Feature Scaling
                """)
                st.markdown("---")  # Horizontal line separator

                # Display final results
                st.subheader("Final Preprocessed Data Preview")
                st.write(processed_data.head())

                # # Make predictions
                # status_text.markdown("Making predictions...")
                # clusters = model.predict(processed_data)
                # processed_data['Cluster'] = clusters

                # # Show shape of data before and after preprocessing
                # st.subheader("Data Shape")
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.write("Original Shape:", data.shape)
                # with col2:
                #     st.write("Processed Shape:", processed_data.shape)

                # # Cluster Analysis
                # st.subheader("Cluster Analysis")
                # col1, col2 = st.columns(2)

                # with col1:
                #     # Cluster distribution
                #     st.write("Cluster Distribution")
                #     fig, ax = plt.subplots()
                #     processed_data['Cluster'].value_counts().plot(kind='bar')
                #     plt.title("Distribution of Clusters")
                #     plt.xlabel("Cluster")
                #     plt.ylabel("Count")
                #     st.pyplot(fig)

                # with col2:
                #     # PCA visualization
                #     st.write("PCA Visualization")
                #     pca = PCA(n_components=2)
                #     pca_result = pca.fit_transform(processed_data.drop('Cluster', axis=1))
                    
                #     fig, ax = plt.subplots()
                #     scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                #                        c=processed_data['Cluster'], cmap='viridis')
                #     plt.colorbar(scatter)
                #     plt.title("Cluster Visualization (PCA)")
                #     plt.xlabel("First Principal Component")
                #     plt.ylabel("Second Principal Component")
                #     st.pyplot(fig)

                # # Cluster Profiles with Age Group Analysis
                # st.subheader("Cluster Profiles")
                # for cluster in processed_data['Cluster'].unique():
                #     with st.expander(f"Cluster {cluster} Profile"):
                #         cluster_data = processed_data[processed_data['Cluster'] == cluster]
                        
                #         # General statistics
                #         col1, col2 = st.columns(2)
                #         with col1:
                #             st.write(f"Size: {len(cluster_data)} customers")
                #             st.write("Key Statistics:")
                #             st.write(cluster_data.describe())
                        
                #         with col2:
                #             if 'Age_Group' in cluster_data.columns:
                #                 st.write("Age Group Distribution:")
                #                 age_dist = cluster_data['Age_Group'].value_counts()
                #                 fig, ax = plt.subplots()
                #                 age_dist.plot(kind='pie', autopct='%1.1f%%')
                #                 plt.title(f"Age Groups in Cluster {cluster}")
                #                 st.pyplot(fig)

            except Exception as e:
                progress_bar.progress(0)
                status_text.markdown(f"❌ **Error occurred during preprocessing:**\n\n{str(e)}")
                st.error(f"An error occurred during preprocessing: {str(e)}")

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