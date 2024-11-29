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

# Add information about the app
with st.expander("About this app"):
    st.write("""
    - This app performs customer segmentation using K-means clustering. 
    - The dataset used in this app is a marketing campaign dataset from a company that sells products to customers.
    - The goal of this app is to help the company to understand their customers better and to develop more effective marketing strategies.
    
    - To use this app, you can either upload your own dataset or use the source data directly.
        - If you choose to upload your own dataset, please ensure that your dataset has the same structure as the source data.
        - If you choose to use the source data directly, you can select the 'Use Source Data Directly' option and it will load the source data from gdrive.
    
    - After selecting the input type, you can follow the preprocessing steps in the sidebar on the left to preprocess your data.
    - You may see some default values for the preprocessing steps, you can keep it or change it but as for now, it is set to the default values.
                      
    - The preprocessing steps are:
        1. Data type conversion
        2. Missing values imputation
        3. Outlier handling
        4. Feature engineering (including Age Group categorization)
        5. Feature encoding
        6. Column dropping
        7. Feature scaling
        8. PCA transformation
    
    - This app will then perform clustering using K-means clustering and visualize the results using PCA and the statistical summary of each cluster.
    
    - Also this app have feature to predict the customer segment for new data based on the trained model.
    - To do so, you can input the customer data and the app will predict the customer segment for the new data.
    """)

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('kmeans_model.joblib')

model = load_model()

# Input type selection
input_type = st.radio('Select Input Type', ['File Upload Local', 'Use Source Data Directly'])
if input_type == 'File Upload Local':
    st.write("You can download and then upload data locally from the source below:")
    st.link_button("Download Source Dataset", "https://drive.google.com/file/d/17JjPC-T_QSACwG29or48HpeJATbCPaDg/view?usp=sharing")

# Initialize data variable
data = None

if input_type == 'File Upload Local':
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
else:
    st.write("You can use the source data directly by selecting the 'Use Source Data Directly' option.")
    try:
        # Link to the file
        file_url = "https://drive.google.com/uc?id=17JjPC-T_QSACwG29or48HpeJATbCPaDg"

        # Load the CSV into a DataFrame
        file_data = pd.read_csv(file_url, index_col=0)

        st.subheader("Raw Data Preview")
        st.write(file_data.head())
            
        # Display data information
        st.subheader("Data Information")
        st.write(check_data_information(file_data, file_data.columns))
            
        # Create a copy for preprocessing
        data = file_data.copy()

    except Exception as e:
        st.error(f"Error loading the file: {e}")       

# Side bar preprocessing steps (only proceed if data is not None)
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
                    original_processed_data = processed_data.copy()
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
                progress_bar.progress(90)
                st.success("Feature scaling completed successfully!")
                st.write("After Feature Scaling:")
                st.write(processed_data.head(3))

                # 8. Transform data using PCA
                pca = PCA(n_components=0.85)  # Keep components that explain percentage of variance
                processed_data_pca = pca.fit_transform(processed_data)
                processed_data_pca = pd.DataFrame(processed_data_pca, columns=[f'PC {i+1}' for i in range(processed_data_pca.shape[1])])
                progress_bar.progress(100)
                st.success("PCA transformation completed successfully!")
                st.subheader("Final Preprocessed Data After PCA")
                st.write(processed_data_pca.head(3))     

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
                8. ✅ PCA Transformation
                """)
                st.markdown("---")  # Horizontal line separator           

            except Exception as e:
                progress_bar.progress(0)
                status_text.markdown(f"❌ **Error occurred during preprocessing:**\n\n{str(e)}")
                st.error(f"An error occurred during preprocessing: {str(e)}")
        
    # Cluster Section    
    st.header("Cluster Section")
    
    # Use the imported model to predict the clusters
    model.fit(processed_data_pca.values)

    # Assign the cluster to our original dataframe and scaled dataframe
    processed_data_pca['Clusters'] = model.labels_
    original_processed_data['Clusters'] = model.labels_

    # Check the original dataframe with assigned cluster for each data
    st.subheader("Data with Assigned Clusters")
    st.write(original_processed_data.head(3))

    # Visualization Section
    st.markdown("---")  # Separator
    st.header("Cluster Visualization")

    # Create the PCA visualization
    fig, ax = plt.subplots(figsize=(10, 6))  # Fixed figure size
    scatter = sns.scatterplot(
        data=processed_data_pca,
        x='PC 1',
        y='PC 2',
        hue='Clusters',
        palette='deep',
        edgecolor='white',
        alpha=0.7,
        s=100  # Fixed point size
    )

    # Improve plot styling
    plt.title('Customer Segments Visualization (PCA)', pad=20, fontsize=14)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)

    # Add a legend with a better position
    plt.legend(title='Clusters', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Make the plot more streamlit-friendly
    st.pyplot(fig, use_container_width=True)

    # Add explanation
    st.markdown("""
    **Understanding the Cluster Plot:**
    - Each point represents a customer
    - Colors indicate different customer segments
    - Closer points suggest similar customer characteristics
    - Distance between clusters shows how distinct the segments are
    """)

    # After the plot explanation, add:
    st.markdown("---")  # Separator
    st.header("Cluster Statistical Summary")

    # Define important features and their aggregation functions
    summary_features = [
        'Total_Acc_Camp', 'NumWebVisitsMonth', 'Income', 
        'Total_Purchases', 'Total_Spending', 'CVR', 'Clusters'
    ]

    agg_funcs = {
        'Total_Acc_Camp': 'sum',
        'NumWebVisitsMonth': 'sum',
        'Income': 'mean',
        'Total_Purchases': 'mean',
        'Total_Spending': ['mean', 'median'],
        'CVR': ['mean', 'median', 'count']
    }

    try:
        # Calculate statistics
        result = round(original_processed_data[summary_features].groupby('Clusters').agg(agg_funcs), 2)
        result.columns = ['_'.join(col).strip() for col in result.columns.values]

        # Rename the count column and add percentage
        result.rename(columns={'CVR_count': 'Cluster Count'}, inplace=True)
        result['Cluster Percentage'] = round((result['Cluster Count'] / result['Cluster Count'].sum()) * 100, 2)

        # Display the results with better formatting
        st.markdown("### Key Metrics by Cluster")
        st.markdown("""
        This table shows important statistics for each cluster:
        - **Campaign Acceptance & Web Visits**: Total accepted campaigns and web visits
        - **Income & Purchases**: Average income and purchase behavior
        - **Spending & Conversion**: Mean/median spending and conversion rates
        - **Cluster Size**: Number and percentage of customers in each cluster
        """)

        # Display the dataframe with styled formatting exactly as in the notebook
        st.dataframe(
            result.style.format({
                'Total_Acc_Camp': '{:.0f}',
                'NumWebVisitsMonth': '{:.0f}',
                'Income_mean': '${:,.2f}',
                'Total_Purchases_mean': '{:.2f}',
                'Total_Spending_mean': '${:,.2f}',
                'Total_Spending_median': '${:,.2f}',
                'CVR_mean': '{:.2%}',
                'CVR_median': '{:.2%}',
                'Cluster Count': '{:.0f}',
                'Cluster Percentage': '{:.1f}%'
            })
        )

    except Exception as e:
        st.error(f"Error in statistical summary: {str(e)}")
        st.write("Available columns in DataFrame:", original_processed_data[summary_features].columns.tolist())

    # After the cluster statistical summary section, add:
    st.markdown("---")
    st.header("Predict New Customer Segment")

    # Use the already uploaded data as reference
    if uploaded_data is not None:
        try:
            st.subheader("Enter Customer Data")
            
            # Create a dictionary to store input values
            prediction_input = {}
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            # Split columns into two groups for layout
            all_columns = file_data.columns.tolist()
            mid_point = len(all_columns) // 2
            
            with col1:
                # First half of columns
                for column in all_columns[:mid_point]:
                    if pd.api.types.is_datetime64_any_dtype(file_data[column]):
                        prediction_input[column] = st.date_input(f'Enter {column}')
                    
                    elif pd.api.types.is_numeric_dtype(file_data[column]):
                        col_min = file_data[column].min()
                        col_max = file_data[column].max()
                        col_mean = file_data[column].mean()
                        
                        prediction_input[column] = st.number_input(
                            f'Enter {column}',
                            min_value=float(col_min),
                            max_value=float(col_max),
                            value=float(col_mean),
                            step=0.1
                        )
                    
                    elif pd.api.types.is_categorical_dtype(file_data[column]) or file_data[column].dtype == 'object':
                        unique_values = file_data[column].unique()
                        prediction_input[column] = st.selectbox(
                            f'Select {column}',
                            options=list(unique_values)
                        )
                    
                    else:
                        prediction_input[column] = st.text_input(f'Enter {column}')
            
            with col2:
                # Second half of columns
                for column in all_columns[mid_point:]:
                    if pd.api.types.is_datetime64_any_dtype(file_data[column]):
                        prediction_input[column] = st.date_input(f'Enter {column}')
                    
                    elif pd.api.types.is_numeric_dtype(file_data[column]):
                        col_min = file_data[column].min()
                        col_max = file_data[column].max()
                        col_mean = file_data[column].mean()
                        
                        prediction_input[column] = st.number_input(
                            f'Enter {column}',
                            min_value=float(col_min),
                            max_value=float(col_max),
                            value=float(col_mean),
                            step=0.1
                        )
                    
                    elif pd.api.types.is_categorical_dtype(file_data[column]) or file_data[column].dtype == 'object':
                        unique_values = file_data[column].unique()
                        prediction_input[column] = st.selectbox(
                            f'Select {column}',
                            options=list(unique_values)
                        )
                    
                    else:
                        prediction_input[column] = st.text_input(f'Enter {column}')

            # Add predict button
            if st.button('Predict Segment'):
                try:
                    # Convert input to DataFrame
                    input_df = pd.DataFrame([prediction_input])
                    
                    # Show the input data
                    st.subheader("Input Data")
                    st.write(input_df)
                    
                    # Create a copy for preprocessing
                    processed_input = input_df.copy()
                    
                    # Apply the same preprocessing steps
                    if 'Dt_Customer' in processed_input.columns:
                        processed_input['Dt_Customer'] = pd.to_datetime(processed_input['Dt_Customer'], format='%Y-%m-%d')
                    
                    # Apply feature engineering
                    feature_engineering(processed_input, middle_age_threshold, senior_age_threshold)
                    
                    # Apply encoding
                    processed_input = feature_encoding(processed_input, columns_to_encode)
                    
                    # Drop unnecessary columns
                    processed_input = drop_columns(processed_input, columns_to_drop)
                    
                    # Apply scaling
                    processed_input = feature_scaling(processed_input)
                    
                    # Apply PCA transformation
                    input_pca = pca.transform(processed_input)
                    
                    # Make prediction
                    cluster = model.predict(input_pca)[0]
                    
                    # Display result with styling
                    st.success(f"### Predicted Customer Segment: {cluster}")
                    
                    # Show cluster characteristics
                    st.subheader(f"Cluster {cluster} Characteristics")
                    cluster_stats = result.loc[cluster]
                    
                    # Display key metrics for the predicted cluster
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Income", f"${cluster_stats['Income_mean']:,.2f}")
                    with col2:
                        st.metric("Average Spending", f"${cluster_stats['Total_Spending_mean']:,.2f}")
                    with col3:
                        st.metric("Conversion Rate", f"{cluster_stats['CVR_mean']:.1%}")
                    
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    st.write("Please ensure all required fields are filled correctly.")
        
        except Exception as e:
            st.error(f"Error setting up prediction form: {str(e)}")
    else:
        st.info("Please upload a data file first to enable prediction")