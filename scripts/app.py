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
import os

# Import all preprocessing functions
from preprocessing import (
    check_data_information,
    change_data_type,
    handle_missing_values,
    filter_outliers,
    feature_engineering,
    feature_encoding,
    drop_columns,
    feature_scaling
)

from cluster_interpretations import get_cluster_interpretations
from feature_definitions import get_feature_definitions

# Page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation Analysis")

# Author Information
st.markdown("""
#### Author
Developed by : Muhammad Cikal Merdeka | Data Analyst/Data Scientist

- [GitHub Profile](https://github.com/mcikalmerdeka)  
- [LinkedIn Profile](https://www.linkedin.com/in/mcikalmerdeka)
""")

# Add information about the app
with st.expander("**Read Instructions First: About This App**"):
    st.write("""
    ## 🎯 Customer Segmentation Application

    ### 📌 Purpose
    - This app performs customer segmentation using K-means clustering
    - Utilizes a marketing campaign dataset from a product-selling company
    - Goal: Help the company understand customers and develop more effective marketing strategies

    ### 🔍 How to Use the App

    #### Data Input Options
    - Two primary ways to load data:
        1. Upload Custom Dataset
            - Ensure your dataset matches the source data structure
        2. Use Source Data Directly
            - Select 'Use Source Data Directly' option
            - Automatically loads data from Google Drive

    ### 🔧 Preprocessing Workflow
    The application follows a comprehensive data preprocessing pipeline:

    1. **Data Type Conversion**
        - Standardize and validate data types

    2. **Missing Values Imputation**
        - Handle and fill missing data points using appropriate strategies

    3. **Outlier Handling**
        - Identify and manage extreme or anomalous data points

    4. **Feature Engineering**
        - Create derived features
        - Categorize variables (e.g., Age Group categorization)

    5. **Feature Encoding**
        - Convert categorical variables to numerical representations

    6. **Column Selection**
        - Retain most relevant features for analysis

    7. **Feature Scaling**
        - Normalize data to ensure balanced analysis

    8. **Dimensionality Reduction**
        - Apply PCA (Principal Component Analysis) for optimization

    ### 🧩 Clustering and Visualization
    - Perform K-means clustering
    - Visualize results using PCA
    - Generate statistical summary for each cluster

    ### 🔮 Prediction Capabilities
    - Predict customer segments for new data
    - Input customer data
    - Receive instant segment classification based on trained model

    ### ⚠️ Important Notes
    - Default preprocessing values are pre-configured
    - Users can modify settings as needed
    - Predictions are based on machine learning model insights
    """)

# Load pre-trained model
@st.cache_resource
def load_model():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root and into models folder
    model_path = os.path.join(script_dir, '..', 'models', 'kmeans_model.joblib')
    return joblib.load(model_path)

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
            
            # Add Data Dictionary section here
            with st.expander("📚 Data Dictionary"):
                st.markdown("### Feature Information")
                
                # Create DataFrame from feature definitions
                definitions = get_feature_definitions()
                feature_df = pd.DataFrame.from_dict(definitions, orient='index')
                
                # Reorder columns and reset index to show feature names as a column
                feature_df = feature_df.reset_index().rename(columns={'index': 'Feature Name'})
                feature_df = feature_df[['Feature Name', 'description', 'data_type', 'specific_type']]
                
                # Rename columns for display
                feature_df.columns = ['Feature Name', 'Description', 'Data Type', 'Specific Type']
                
                # Display as a styled table
                st.dataframe(
                    feature_df.style.set_properties(**{
                        'background-color': 'white',
                        'color': 'black',
                        'border-color': 'lightgrey'
                    })
                )
                
                st.markdown("""
                **Note:**
                - Categorical (Nominal): Categories without any natural order
                - Categorical (Ordinal): Categories with a natural order
                - Numerical (Discrete): Whole numbers
                - Numerical (Continuous): Any numerical value
                """)

            # Create a copy for preprocessing
            data = file_data.copy()

        except Exception as e:
            st.error(f"Error loading the file: {e}")
else:
    st.write("You can use the source data directly by selecting the 'Use Source Data Directly' option.")
    try:
        # # Link to the file (using gdrive link)
        # file_url = "https://drive.google.com/uc?id=17JjPC-T_QSACwG29or48HpeJATbCPaDg"

        # Link to the file (using github link)
        file_url = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/main/marketing_campaign_data.csv"

        # Load the CSV into a DataFrame
        file_data = pd.read_csv(file_url, index_col=0)

        st.subheader("Raw Data Preview")
        st.write(file_data.head())
            
        # Display data information
        st.subheader("Data Information")
        st.write(check_data_information(file_data, file_data.columns))
        
        # Add Data Dictionary section here
        with st.expander("📚 Data Dictionary"):
            st.markdown("### Feature Information")
            
            # Create DataFrame from feature definitions
            definitions = get_feature_definitions()
            feature_df = pd.DataFrame.from_dict(definitions, orient='index')
            
            # Reorder columns and reset index to show feature names as a column
            feature_df = feature_df.reset_index().rename(columns={'index': 'Feature Name'})
            feature_df = feature_df[['Feature Name', 'description', 'data_type', 'specific_type']]
            
            # Rename columns for display
            feature_df.columns = ['Feature Name', 'Description', 'Data Type', 'Specific Type']
            
            # Display as a styled table
            st.dataframe(
                feature_df.style.set_properties(**{
                    'background-color': 'white',
                    'color': 'black',
                    'border-color': 'lightgrey'
                })
            )
            
            st.markdown("""
            **Note:**
            - Categorical (Nominal): Categories without any natural order
            - Categorical (Ordinal): Categories with a natural order
            - Numerical (Discrete): Whole numbers
            - Numerical (Continuous): Any numerical value
            """)

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

                # Store the date converted processed data in session state
                date_converted_data = processed_data.copy()
                st.session_state['date_converted_data'] = date_converted_data
                
                # 2. Impute missing values
                status_text.markdown("**Step 2/7:** Imputing missing values...")
                if columns_to_impute:
                    processed_data[columns_to_impute] = handle_missing_values(processed_data, columns_to_impute)
                progress_bar.progress(28)
                st.success("Missing value imputation completed successfully!")
                st.write("After Missing Value Imputation:")
                st.write(processed_data.head(3))

                # Store the imputed processed data in session state
                imputed_processed_data = processed_data.copy()
                st.session_state['imputed_processed_data'] = imputed_processed_data
                
                # 3. Handle outliers
                status_text.markdown("**Step 3/7:** Handling outliers...")
                if outlier_columns:
                    processed_data = filter_outliers(processed_data, outlier_columns)
                progress_bar.progress(42)
                st.success("Outlier handling completed successfully!")
                st.write("After Outlier Handling:")
                st.write(processed_data.head(3))

                # Store the outlier handled processed data in session state
                outlier_handled_processed_data = processed_data.copy()
                st.session_state['outlier_handled_processed_data'] = outlier_handled_processed_data
                
                # 4. Feature engineering
                status_text.markdown("**Step 4/7:** Performing feature engineering...")
                if do_feature_engineering:
                    feature_engineering(processed_data, middle_age_threshold, senior_age_threshold)
                progress_bar.progress(56)
                st.success("Feature engineering completed successfully!")
                st.write("After Feature Engineering:")
                st.write(processed_data.head(3))

                # Store the feature engineered processed data in session state
                feature_engineered_processed_data = processed_data.copy()
                st.session_state['feature_engineered_processed_data'] = feature_engineered_processed_data
                
                # 5. Categorical encoding
                status_text.markdown("**Step 5/7:** Encoding categorical features...")
                if columns_to_encode:
                    # Use processed_data for encoding
                    st.write("Columns to encode:", columns_to_encode)
                    # st.write("Available columns before encoding:", processed_data.columns.tolist())
                    processed_data = feature_encoding(processed_data, columns_to_encode)
                progress_bar.progress(70)
                st.success("Categorical encoding completed successfully!")
                st.write("After Categorical Encoding:")
                st.write(processed_data.head(3))

                # Store the encoded processed data in session state
                encoded_processed_data = processed_data.copy()
                st.session_state['encoded_processed_data'] = encoded_processed_data
                
                # 6. Drop columns
                status_text.markdown("**Step 6/7:** Dropping selected columns...")
                if columns_to_drop:
                    processed_data = drop_columns(processed_data, columns_to_drop)
                progress_bar.progress(84)
                st.success("Column dropping completed successfully!")
                st.write("After Dropping Columns:")
                st.write(processed_data.head(3))

                # Store the dropped columns processed data in session state
                dropped_columns_processed_data = processed_data.copy()
                st.session_state['dropped_columns_processed_data'] = dropped_columns_processed_data
                
                # 7. Feature scaling
                status_text.markdown("**Step 7/7:** Scaling features...")
                processed_data, scalers = feature_scaling(processed_data)
                progress_bar.progress(90)
                st.success("Feature scaling completed successfully!")
                st.write("After Feature Scaling:")
                st.write(processed_data.head(3))

                # Store the scaled processed data and original data in session state
                scaled_processed_data = processed_data.copy()
                st.session_state['scaled_processed_data'] = scaled_processed_data
                st.session_state['original_processed_data'] = processed_data.copy()
                st.session_state['scalers'] = scalers

                # 8. Transform data using PCA
                pca = PCA(n_components=0.85)
                processed_data_pca = pca.fit_transform(processed_data)
                processed_data_pca = pd.DataFrame(processed_data_pca, columns=[f'PC {i+1}' for i in range(processed_data_pca.shape[1])])
                progress_bar.progress(100)
                st.success("PCA transformation completed successfully!")
                st.subheader("Final Preprocessed Data After PCA")
                st.write(processed_data_pca.head(3))

                # Store the PCA transformed processed data in session state
                st.session_state['processed_data_pca'] = processed_data_pca

                # Store preprocessing parameters in session state
                st.session_state['preprocessing_params'] = {
                    'date_columns': date_columns,
                    'date_format': date_format,
                    'middle_age_threshold': middle_age_threshold,
                    'senior_age_threshold': senior_age_threshold,
                    'columns_to_encode': columns_to_encode,
                    'columns_to_drop': columns_to_drop,
                    # Store the unique categories for each categorical column
                    'categorical_mappings': {
                        col: st.session_state['feature_engineered_processed_data'][col].unique().tolist() 
                        for col in columns_to_encode 
                        if col in st.session_state['feature_engineered_processed_data'].columns
                    },
                    # Store the scaling parameters (mean and std) for numerical columns
                    'scaling_params': {
                        'mean': processed_data.mean().to_dict(),
                        'std': processed_data.std().to_dict()
                    }
                }

                # Store the PCA and model
                st.session_state['pca'] = pca
                st.session_state['model'] = model
                
                # Fit the model here
                model.fit(processed_data_pca.values)
                
                # Store clusters in session state BEFORE using them
                clusters = model.labels_
                st.session_state['clusters'] = clusters

                # Add completion message and separator
                st.markdown("---")
                st.success("🎉 All preprocessing steps completed successfully!")
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
        
    # Cluster Section - Only show if preprocessing has been done AND clusters exist
    if 'processed_data_pca' in st.session_state and 'clusters' in st.session_state:
        st.header("Cluster Section")
        
        processed_data_pca = st.session_state['processed_data_pca']
        original_processed_data = st.session_state['feature_engineered_processed_data']
        
        # Assign the cluster to our original dataframe and scaled dataframe
        processed_data_pca['Clusters'] = st.session_state['clusters']
        original_processed_data['Clusters'] = st.session_state['clusters']

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

            # Display the dataframe statistical summary
            st.dataframe(
                result.style.format({
                    'Total_Acc_Camp': '{:.0f}',
                    'NumWebVisitsMonth': '{:.0f}',
                    'Income_mean': '${:,.2f}',
                    'Total_Purchases_mean': '{:.2f}',
                    'Total_Spending_mean': '${:,.2f}',
                    'Total_Spending_median': '${:,.2f}',
                    'CVR_mean (%)': '{:.2f}%',
                    'CVR_median (%)': '{:.2f}%',
                    'Cluster Count': '{:.0f}',
                    'Cluster Percentage': '{:.1f}%'
                })
            )

        except Exception as e:
            st.error(f"Error in statistical summary: {str(e)}")
            st.write("Available columns in DataFrame:", original_processed_data[summary_features].columns.tolist())

        if 'clusters' in st.session_state:
            st.markdown("---")
            st.header("Cluster Analysis and Recommendations")
            
            interpretations = get_cluster_interpretations()
            
            # Create tabs for different aspects of the interpretation
            tab1, tab2 = st.tabs(["Cluster Details", "Business Recommendations"])
            
            with tab1:
                # Display cluster interpretations
                for cluster_id, cluster_info in interpretations["clusters"].items():
                    with st.expander(f"Cluster {cluster_id}: {cluster_info['name']} ({cluster_info['percentage']})"):
                        st.markdown("### Description")
                        st.markdown(cluster_info['description'])
            
            with tab2:
                st.subheader("Business Recommendations")
                
                # Display cluster-specific recommendations
                for cluster_id, cluster_info in interpretations["clusters"].items():
                    with st.expander(f"Cluster {cluster_id}: {cluster_info['name']} ({cluster_info['percentage']})"):
                        for rec_title, rec_desc in cluster_info['recommendations'].items():
                            st.markdown(f"**{rec_title}**")
                            st.markdown(f"<br>{rec_desc}", unsafe_allow_html=True)
                
                # Display cross-cluster initiatives
                with st.expander("Cross-Cluster Initiatives"):
                    for initiative_title, initiative_desc in interpretations["cross_cluster_initiatives"].items():
                        st.markdown(f"**{initiative_title}**")
                        st.markdown(f"<br>{initiative_desc}", unsafe_allow_html=True)



## Predict New Customer Segment Section
if 'pca' in st.session_state:  # Only show prediction section if preprocessing is done
    st.markdown("---")
    st.header("Predict New Customer Segment")

    # Use the already uploaded data as reference
    if data is not None:
        try:
            st.subheader("Enter Customer Data")
            
            with st.form("prediction_form"):
                # Create a dictionary to store input values
                prediction_input = {}
                
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                
                # Split columns into two groups for layout
                all_columns = file_data.columns.tolist()
                mid_point = len(all_columns) // 2

                # Initial convert Dt_Customer to datetime
                file_data['Dt_Customer'] = pd.to_datetime(file_data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
                
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
                                f"Enter {column}",
                                min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                                max_value=float(col_max) if not pd.isna(col_max) else None,
                                value=float(col_mean) if not pd.isna(col_mean) else 0.0,
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
                                f"Enter {column}",
                                min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                                max_value=float(col_max) if not pd.isna(col_max) else None,
                                value=float(col_mean) if not pd.isna(col_mean) else 0.0,
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

                # Submit button
                submit_button = st.form_submit_button("Predict Segment")

            # Move the prediction logic outside the form but check for the submit button
            if submit_button:
                try:
                    # Convert input to DataFrame
                    input_df = pd.DataFrame([prediction_input])
                    
                    # Show the input data
                    st.subheader("New Customer Input Data")
                    st.write(input_df)
                    
                    # Create a copy for preprocessing
                    processed_input = input_df.copy()

                    # Get preprocessing parameters
                    params = st.session_state['preprocessing_params']
                    
                    # 1. Handle dates
                    for date_col in params['date_columns']:
                        if date_col in processed_input.columns:
                            processed_input[date_col] = pd.to_datetime(processed_input[date_col], format=params['date_format'])

                    # 2. Feature engineering
                    if 'Dt_Customer' in processed_input.columns:
                        current_year = datetime.now().year
                        
                        # Age calculation
                        processed_input['Age'] = current_year - processed_input['Year_Birth']
                        
                        # Age group
                        def custom_age_group(x):
                            if x >= params['senior_age_threshold']:
                                return 'Senior Adult'
                            elif x >= params['middle_age_threshold']:
                                return 'Middle Adult'
                            else:
                                return 'Young Adult'
                        
                        processed_input['Age_Group'] = processed_input['Age'].apply(custom_age_group)
                        
                        # Membership Duration
                        processed_input['Membership_Duration'] = current_year - processed_input['Dt_Customer'].dt.year
                        
                        # Total accepted campaigns
                        campaign_cols = [col for col in processed_input.columns if 'AcceptedCmp' in col]
                        processed_input['Total_Acc_Camp'] = processed_input[campaign_cols].sum(axis=1).astype('int64')
                        
                        # Total spending
                        spending_cols = ['MntCoke', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
                        processed_input['Total_Spending'] = processed_input[spending_cols].sum(axis=1).astype('int64')
                        
                        # Total purchases
                        purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
                        processed_input['Total_Purchases'] = processed_input[purchase_cols].sum(axis=1).astype('int64')
                        
                        # Conversion Rate (CVR)
                        processed_input['CVR'] = np.round(processed_input['Total_Purchases'] / processed_input['NumWebVisitsMonth'], 2)
                        processed_input['CVR'].fillna(0, inplace=True)
                        processed_input['CVR'].replace([np.inf, -np.inf], 0, inplace=True)

                    st.subheader("After Feature Engineering")
                    st.write(processed_input)

                    # 3. Feature encoding
                    try:
                        # Handle ordinal encoding for Education
                        if 'Education' in processed_input.columns:
                            degree_order = ['SMA', 'D3', 'S1', 'S2', 'S3']
                            education_map = {deg: idx for idx, deg in enumerate(degree_order)}
                            processed_input['Education'] = processed_input['Education'].map(education_map)

                        # Handle ordinal encoding for Age_Group
                        if 'Age_Group' in processed_input.columns:
                            age_group_order = ['Young Adult', 'Middle Adult', 'Senior Adult']
                            age_group_map = {group: idx for idx, group in enumerate(age_group_order)}
                            processed_input['Age_Group'] = processed_input['Age_Group'].map(age_group_map)

                        # Handle one-hot encoding for Marital_Status with expected dummy columns
                        if 'Marital_Status' in processed_input.columns:
                            marital_dummies = pd.get_dummies(processed_input['Marital_Status'], prefix='Marital_Status', drop_first=True)
                            
                            # Drop original column and add encoded columns
                            processed_input = processed_input.drop(columns=['Marital_Status'], errors='ignore')
                            processed_input = pd.concat([processed_input, marital_dummies], axis=1)

                        # Ensure all expected columns are present before moving to scaling
                        expected_columns = st.session_state['dropped_columns_processed_data'].columns.tolist()
                        for col in expected_columns:
                            if col not in processed_input.columns:
                                processed_input[col] = 0

                        for col in processed_input.columns:
                            if col not in expected_columns:
                                processed_input.drop(columns=col, inplace=True)

                        # Reorder and match columns to match training data
                        processed_input = processed_input[expected_columns]

                    except Exception as e:
                        st.error(f"Error in feature encoding: {str(e)}")
                        st.write("Debug information:")
                        st.write("Current columns:", processed_input.columns.tolist())
                        st.write("Expected columns:", expected_columns)

                    st.subheader("After Feature Encoding and Drop Columns")
                    st.write(processed_input)

                    # 4. Feature scaling
                    try:
                        # Get the features that each scaler was trained on
                        standard_features = st.session_state['scalers']['standard'].feature_names_in_.tolist()
                        
                        # # Debug information
                        # st.write("Columns before scaling:", processed_input.columns.tolist())
                        # st.write("Standard scaler features:", standard_features)
                        
                        # Create separate DataFrames for each scaling type
                        standard_df = processed_input[standard_features].copy()
                        
                        # Apply standard scaling only to features that were in the training data
                        processed_input[standard_features] = st.session_state['scalers']['standard'].transform(standard_df)
                        
                        # Log transform specific features (without scaling)
                        log_transform_features = [
                            'MntCoke', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                            'MntSweetProducts', 'MntGoldProds', 'Total_Spending', 'CVR'
                        ]
                        for feature in log_transform_features:
                            if feature in processed_input.columns:
                                processed_input[feature] = np.log1p(processed_input[feature])
                        
                        # MinMax scale specific features
                        count_features = [
                            'NumWebVisitsMonth', 'NumDealsPurchases', 'NumWebPurchases', 
                            'NumCatalogPurchases', 'NumStorePurchases', 'Total_Purchases'
                        ]
                        if 'minmax' in st.session_state['scalers']:
                            count_features_available = [f for f in count_features if f in processed_input.columns]
                            if count_features_available:
                                count_df = processed_input[count_features_available].copy()
                                processed_input[count_features_available] = st.session_state['scalers']['minmax'].transform(count_df)

                    except Exception as e:
                        st.error(f"Error in feature scaling: {str(e)}")
                        st.write("Available columns:", processed_input.columns.tolist())
                        st.write("Scaler features:", st.session_state['scalers']['standard'].feature_names_in_.tolist())

                    st.subheader("After Feature Scaling")
                    st.write(processed_input)

                    # 5. Apply PCA transformation
                    try:
                        input_pca = st.session_state['pca'].transform(processed_input)
                    except Exception as e:
                        st.error(f"Error in PCA transformation: {str(e)}")
                        st.write("Available columns:", processed_input.columns.tolist())

                    st.subheader("After PCA Transformation")
                    st.write(input_pca)

                    # 6. Make prediction
                    try:
                        cluster = st.session_state['model'].predict(input_pca)[0]
                        st.success(f"### Predicted Customer Segment: {cluster}")

                        # Create visualization of the prediction
                        st.subheader("Customer Segment Visualization")
                        
                        # Get the existing PCA data
                        existing_data = st.session_state['processed_data_pca'].copy()
                        existing_clusters = st.session_state['clusters']
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot existing clusters
                        scatter = sns.scatterplot(
                            data=existing_data,
                            x='PC 1',
                            y='PC 2',
                            hue=existing_clusters,
                            palette='deep',
                            alpha=0.6,
                            s=100,
                            legend='brief'
                        )
                        
                        # Plot the new customer point
                        plt.scatter(
                            input_pca[0][0],  # First PC
                            input_pca[0][1],  # Second PC
                            c='red',          # Red color for new point
                            marker='*',       # Star marker
                            s=300,            # Larger size
                            label='New Customer'
                        )
                        
                        # Improve plot styling
                        plt.title('Customer Segment Prediction Visualization', pad=20, fontsize=14)
                        plt.xlabel('First Principal Component', fontsize=12)
                        plt.ylabel('Second Principal Component', fontsize=12)
                        
                        # Add legend
                        handles, labels = plt.gca().get_legend_handles_labels()
                        plt.legend(
                            handles=handles, 
                            labels=['Cluster ' + str(i) for i in range(len(set(existing_clusters)))] + ['New Customer'],
                            title='Segments',
                            title_fontsize=12,
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left'
                        )
                        
                        # Display the plot
                        st.pyplot(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        **Understanding the Visualization:**
                        - Colored dots represent existing customer segments
                        - Red star (★) shows where the new customer fits
                        - Closer points indicate similar characteristics
                        - Position relative to clusters shows segment alignment
                        """)

                    except Exception as e:
                        st.error(f"An error occurred during prediction visualization: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")      

        except Exception as e:
            st.error(f"Error setting up prediction form: {str(e)}")
    else:
        st.info("Please upload a data file first to enable prediction")
else:
    st.info("Please complete the preprocessing steps first before making predictions.")