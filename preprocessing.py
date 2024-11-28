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

## Additional : Data information function
# Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df

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
def feature_engineering(data, middle_age_threshold=40, senior_age_threshold=60):
    """
    Engineer features from existing data
    
    Parameters:
    data: DataFrame to process
    middle_age_threshold: Age threshold for Middle Adult category (default 40)
    senior_age_threshold: Age threshold for Senior Adult category (default 60)
    """
    # Age
    year_data = datetime.now().year
    data['Age'] = year_data - data['Year_Birth']

    # Age_Group with custom thresholds
    def custom_age_group(x):
        if x >= senior_age_threshold:
            return 'Senior Adult'
        elif x >= middle_age_threshold:
            return 'Middle Adult'
        else:
            return 'Young Adult'

    data['Age_Group'] = data['Age'].apply(custom_age_group)

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
def feature_encoding(data, columns_to_encode):
    """
    Encode categorical features flexibly based on input columns
    """
    # Create a copy of the input DataFrame
    df_preprocessed = data.copy()
    
    if not columns_to_encode:  # If no columns to encode, return original data
        return df_preprocessed
    
    try:
        # Debug print
        print("Available columns:", df_preprocessed.columns.tolist())
        print("Columns to encode:", columns_to_encode)
        
        # Validate columns exist in DataFrame
        missing_columns = [col for col in columns_to_encode if col not in df_preprocessed.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in DataFrame")
        
        # Store original data types
        original_dtypes = df_preprocessed.dtypes
        
        # Identify datetime columns
        datetime_columns = df_preprocessed.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Store datetime data separately
        datetime_data = df_preprocessed[datetime_columns].copy() if datetime_columns else None
        
        # Create transformers list
        transformers = []
        columns_to_encode_copy = columns_to_encode.copy()  # Create a copy to modify
        
        # Special handling for known columns
        if 'Education' in columns_to_encode_copy:
            if 'Education' in df_preprocessed.columns:  # Verify column exists
                degree_order = ['SMA', 'D3', 'S1', 'S2', 'S3']
                transformers.append(
                    ('education', OrdinalEncoder(categories=[degree_order], dtype=np.float64), ['Education'])
                )
                columns_to_encode_copy.remove('Education')
        
        if 'Age_Group' in columns_to_encode_copy:
            if 'Age_Group' in df_preprocessed.columns:  # Verify column exists
                age_group_order = ['Young Adult', 'Middle Adult', 'Senior Adult']
                transformers.append(
                    ('age_group', OrdinalEncoder(categories=[age_group_order], dtype=np.float64), ['Age_Group'])
                )
                columns_to_encode_copy.remove('Age_Group')
        
        # Handle Marital_Status specifically
        if 'Marital_Status' in columns_to_encode_copy:
            if 'Marital_Status' in df_preprocessed.columns:  # Verify column exists
                transformers.append(
                    ('marital_status', OneHotEncoder(sparse_output=False, drop='first', dtype=np.float64), ['Marital_Status'])
                )
                columns_to_encode_copy.remove('Marital_Status')
        
        # Handle remaining categorical columns
        remaining_columns = [col for col in columns_to_encode_copy if col in df_preprocessed.columns]
        if remaining_columns:
            transformers.append(
                ('categorical', OneHotEncoder(sparse_output=False, drop='first', dtype=np.float64), remaining_columns)
            )
        
        if not transformers:  # If no transformers were created
            return df_preprocessed
        
        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Get list of columns being transformed
        columns_being_transformed = sum([cols for _, _, cols in transformers], [])
        
        # Get remaining columns
        remaining_columns = [col for col in df_preprocessed.columns if col not in columns_being_transformed]
        
        # Debug print
        print("Columns being transformed:", columns_being_transformed)
        print("Remaining columns:", remaining_columns)
        
        # Transform the data
        encoded_array = preprocessor.fit_transform(df_preprocessed)
        
        # Get feature names
        feature_names = []
        
        # Add feature names for each transformer
        for name, transformer, columns in transformers:
            if isinstance(transformer, OrdinalEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, OneHotEncoder):
                for i, col in enumerate(columns):
                    cats = preprocessor.named_transformers_[name].categories_[i][1:]
                    feature_names.extend([f"{col}_{cat}" for cat in cats])
        
        # Add remaining column names
        feature_names.extend(remaining_columns)
        
        # Debug print
        print("Feature names:", feature_names)
        print("Encoded array shape:", encoded_array.shape)
        
        # Create DataFrame with encoded features
        df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df_preprocessed.index)
        
        return df_encoded
        
    except Exception as e:
        print(f"Error in feature encoding: {str(e)}")
        print(f"Problematic columns: {columns_to_encode}")
        raise e

## Feature scaling function
def feature_scaling(data):
    """
    Scale features using appropriate scaling methods based on their distributions.
    Applies log transformation to heavily skewed features before scaling.
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
    
    # Convert all numeric columns to float before transformation
    numeric_columns = df_preprocessed.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        df_preprocessed[col] = df_preprocessed[col].astype(float)
    
    # Apply log transformation and scale
    if log_transform_features:
        available_log_features = [col for col in log_transform_features if col in df_preprocessed.columns]
        if available_log_features:
            print(f"Applying log transformation to: {available_log_features}")
            df_preprocessed[available_log_features] = df_preprocessed[available_log_features].astype(float)
            for feature in available_log_features:
                df_preprocessed[feature] = np.log1p(df_preprocessed[feature])
            df_preprocessed[available_log_features] = standard_scaler.fit_transform(df_preprocessed[available_log_features])

    # Scale skewed outlier features
    if skewed_outliers_features:
        available_features = [col for col in skewed_outliers_features if col in df_preprocessed.columns]
        if available_features:
            print(f"Applying robust scaling to: {available_features}")
            df_preprocessed[available_features] = df_preprocessed[available_features].astype(float)
            df_preprocessed[available_features] = robust_scaler.fit_transform(df_preprocessed[available_features])

    # Scale count-based features
    if count_features:
        available_features = [col for col in count_features if col in df_preprocessed.columns]
        if available_features:
            print(f"Applying minmax scaling to: {available_features}")
            df_preprocessed[available_features] = df_preprocessed[available_features].astype(float)
            df_preprocessed[available_features] = minmax_scaler.fit_transform(df_preprocessed[available_features])

    # Scale normally distributed features
    if standard_features:
        available_features = [col for col in standard_features if col in df_preprocessed.columns]
        if available_features:
            print(f"Applying standard scaling to: {available_features}")
            df_preprocessed[available_features] = df_preprocessed[available_features].astype(float)
            df_preprocessed[available_features] = standard_scaler.fit_transform(df_preprocessed[available_features])

    return df_preprocessed