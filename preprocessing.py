# preprocessing.py
import numpy as np
import pandas as pd
from datetime import datetime

# Feature encoding libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Feature scaling libraries
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# =====================================================================Functions for data pre-processing========================================================================

## Date conversion function
def change_data_type(data, column, target_type, format=None):
    if target_type == 'datetime':
        data[column] = pd.to_datetime(data[column], format=format, errors='coerce')
    else:
        data[column] = data[column].astype(target_type, errors='ignore')
    return data

## Impute missing values function
def impute_missing_values(data, columns):
    return data[columns].fillna(data[columns].median())

## Drop columns function
def drop_columns(data, columns):
    return data.drop(columns=columns)

## Handle outliers function
def filter_outliers(data, col_series):
    # Boolean mask for rows without outliers
    filtered_entries = np.array([True] * len(data))
    
    for col in col_series:
        Q1 = data[col].quantile(0.25)  
        Q3 = data[col].quantile(0.75)  
        IQR = Q3 - Q1  
        lower_bound = Q1 - (IQR * 1.5)  
        upper_bound = Q3 + (IQR * 1.5)  

        filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))

        # Update the mask for filtering rows within bounds for the column
        filtered_entries = filtered_entries & filter_outlier
    
    return data[filtered_entries]

## Feature engineering function
def feature_engineering(data):

    # Age
    year_data = datetime.now().year
    data['Age'] = year_data - data['Year_Birth']

    # Age_Group
    def age_group(x):
        if x >= 60:
            return 'Senior Adult'
        elif x >= 40:
            return 'Middle Adult'
        else:
            return 'Young Adult'

    data['Age_Group'] = data['Age'].apply(age_group)

    # Membership_Duration
    data['Membership_Duration'] = year_data - data['Dt_Customer'].dt.year

    # Total_Acc_Camp
    data['Total_Acc_Camp'] = data.filter(like='AcceptedCmp').sum(axis=1).astype("int64")
    
    # Total_Spending
    data['Total_Spending'] = data.loc[:, ['MntCoke', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']] \
                                 .sum(axis=1) \
                                 .astype('int64')
    
    # Total_Purchases
    data['Total_Purchases'] = data.loc[:, ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']] \
                                  .sum(axis=1) \
                                  .astype('int64')
    
    # Conversion Rate (CVR)
    data['CVR'] = np.round(data['Total_Purchases'] / data['NumWebVisitsMonth'], 2)
    data['CVR'].fillna(0, inplace=True)
    data['CVR'].replace([np.inf, -np.inf], 0, inplace=True)


## Feature encoding function
def feature_encoding(data, degree_order=None, age_group_order=None, marital_status_order=None):
    # Create a copy of the input DataFrame to avoid modifying the original
    df_preprocessed = data.copy()
    
    # Set default orders if not provided
    if degree_order is None:
        degree_order = ['SMA', 'D3', 'S1', 'S2', 'S3']
    if age_group_order is None:
        age_group_order = ['Young Adult', 'Middle Adult', 'Senior Adult']
    if marital_status_order is None:
        marital_status_order = df_preprocessed['Marital_Status'].unique()

    # Store original data types before preprocessing
    original_dtypes = df_preprocessed.dtypes

    # First, identify datetime columns
    datetime_columns = df_preprocessed.select_dtypes(include=['datetime64']).columns.tolist()

    # Store datetime columns separately before preprocessing
    datetime_data = df_preprocessed[datetime_columns].copy() if datetime_columns else None

    # Check if there are any unexpected categories in the data
    unique_degrees = df_preprocessed['Education'].unique()
    unique_age_groups = df_preprocessed['Age_Group'].unique()
    unique_marital = df_preprocessed['Marital_Status'].unique()

    # Validation checks
    if not all(deg in degree_order for deg in unique_degrees):
        print("Warning: Some education degrees in the data are not in the specified order list")
    if not all(age in age_group_order for age in unique_age_groups):
        print("Warning: Some age groups in the data are not in the specified order list")
    if not all(status in marital_status_order for status in unique_marital):
        print("Warning: Some marital status categories in the data are not in the specified order list")

    # Create the column transformer for each pre-processing steps
    preprocessor = ColumnTransformer(
        transformers=[
            # For OrdinalEncoder, the categories parameter needs to be a list of lists or wrapped in double brackets to convert it to a 2D array, where each inner list contains the categories for each feature
            ('education', OrdinalEncoder(categories=[degree_order], dtype=np.float64), ['Education']),
            ('age_group', OrdinalEncoder(categories=[age_group_order], dtype=np.float64), ['Age_Group']),
            
            # drop='first': Drops the first column to avoid the dummy variable trap (optional, depending on whether you want all categories or not).
            # sparse_output=False: This argument controls whether the encoder outputs a sparse matrix. Setting it to False returns a dense array instead of a sparse matrix.
            # dtype=np.float64 : This ensures the transformers output numeric data instead of strings
            ('marital_status', OneHotEncoder(drop='first', sparse_output=False, dtype=np.float64), ['Marital_Status'])
        ],
        # remainder='passthrough' ensures that other columns that are not specified for transformation remain unchanged.
        remainder='passthrough'
    )

    # Remove datetime columns before transformation
    df_for_transform = df_preprocessed.drop(columns=datetime_columns) if datetime_columns else df_preprocessed

    # Apply the transformer
    df_encoded = preprocessor.fit_transform(df_for_transform)

    # Get the non-transformed column names
    passthrough_features = list(df_for_transform.columns.drop(['Education', 'Age_Group', 'Marital_Status']))

    # Create the list of column names
    transformed_features = (
        ['Education', 'Age_Group'] +  # Ordinal encoded columns
        [f'Marital_Status_{cat}' for cat in marital_status_order[1:]] + # One-hot encoded columns (excluding first category)
        passthrough_features  # Remaining columns
    )

    # Convert the result back to a DataFrame
    df_encoded = pd.DataFrame(df_encoded, columns=transformed_features)

    # Convert all columns to float64 except those that should remain as other types
    numeric_columns = df_encoded.columns
    df_encoded[numeric_columns] = df_encoded[numeric_columns].astype(np.float64)

    # Add back datetime columns if they exist
    if datetime_columns:
        for col in datetime_columns:
            df_encoded[col] = datetime_data[col]

    # Define columns that should stay as float64 (encoded columns)
    encoded_columns = ['Education', 'Age_Group'] + [col for col in df_encoded.columns if col.startswith('Marital_Status_')]

    # Reapply original data types for non-encoded columns
    for col in df_encoded.columns:
        if col in original_dtypes and col not in encoded_columns:
            try:
                df_encoded[col] = df_encoded[col].astype(original_dtypes[col])
            except Exception as e:
                print(f"Could not convert column {col} back to {original_dtypes[col]}. Error: {e}")

    return df_encoded

## Feature scaling function
def feature_scaling(data):
    """
    Scale features using appropriate scaling methods based on their distributions.
    Applies log transformation to heavily skewed features before scaling.
    
    Parameters:
    df: Input DataFrame with features to be scaled
    
    Returns:
    DataFrame: Scaled features in a DataFrame with the same structure
    dict: Fitted scalers for future use
    """
    # Create copies to avoid modifying original data
    df_preprocessed = data.copy()
    
    # Initialize scalers
    robust_scaler = RobustScaler(quantile_range=(5, 95))
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Features that need log transformation (heavily skewed even after outlier handling)
    log_transform_features = [
        'MntCoke',
        'MntFruits',
        'MntMeatProducts',
        'MntFishProducts',
        'MntSweetProducts',
        'MntGoldProds',
        'Total_Spending',
        'CVR'
    ]
    
    # Other monetary features (highly skewed with long tails) that might not need log transform (if outliers handled) - RobustScaler
    skewed_outliers_features = []
    
    # Count features (discrete but representing actual quantities) - MinMaxScaler
    count_features = [
        'NumWebVisitsMonth',
        'NumDealsPurchases',
        'NumWebPurchases',
        'NumCatalogPurchases',
        'NumStorePurchases',
        'Total_Purchases'
    ]
    
    # Features with more normal-like distributions - StandardScaler
    standard_features = [
        'Income',
        'Age',
        'Recency',
        'Membership_Duration'  # Though multimodal, it represents actual time periods
    ]
    
    ## Features to skip scaling:
    # Binary categorical (0/1):
    # - AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5
    # - Response
    # - Complain
    # Multi-modal categorical/ordinal/nominal:
    # - Marital_Status one-hot encoded columns (Scaling would distort their meaning as dummy variables)
    # - Education
    # - Age_Group
    # - Kidhome, Teenhome, Num_Child (Small discrete values 0,1,2,3 representing counts, scaling might unnecessarily distort their interpretability in the clustering process)

    # Apply log transformation and scale
    if log_transform_features:
        for feature in log_transform_features:
            if feature in df_preprocessed.columns:
                df_preprocessed[feature] = np.log1p(df_preprocessed[feature])
        df_preprocessed[log_transform_features] = standard_scaler.fit_transform(df_preprocessed[log_transform_features])

    # Scale skewed outlier features
    if skewed_outliers_features:
        df_preprocessed[skewed_outliers_features] = robust_scaler.fit_transform(df_preprocessed[skewed_outliers_features])

    # Scale count-based features
    if count_features:
        df_preprocessed[count_features] = minmax_scaler.fit_transform(df_preprocessed[count_features])

    # Scale normally distributed features
    if standard_features:
        df_preprocessed[standard_features] = standard_scaler.fit_transform(df_preprocessed[standard_features])

    return df_preprocessed