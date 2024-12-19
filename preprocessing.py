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
def handle_missing_values(data, columns, strategy='fill', imputation_method='median'):
    if strategy == 'fill':
        if imputation_method == 'median':
            return data[columns].fillna(data[columns].median())
        elif imputation_method == 'mean':
            return data[columns].fillna(data[columns].mean())
        elif imputation_method == 'mode':
            return data[columns].fillna(data[columns].mode().iloc[0])
        elif imputation_method == 'ffill':
            return data[columns].fillna(method='ffill')
        elif imputation_method == 'bfill':
            return data[columns].fillna(method='bfill')
        else:
            return data[columns].fillna(data[columns].median())

    elif strategy == 'remove':
        return data.dropna(subset=columns)

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
def feature_encoding(data, columns_to_encode, training_mode=True):
    """
    Encode categorical features flexibly based on input columns
    """
    df_preprocessed = data.copy()
    
    if not columns_to_encode:
        return df_preprocessed
    
    try:
        # Handle Education if present
        if 'Education' in columns_to_encode and 'Education' in df_preprocessed.columns:
            degree_order = ['SMA', 'D3', 'S1', 'S2', 'S3']
            education_map = {deg: idx for idx, deg in enumerate(degree_order)}
            df_preprocessed['Education'] = df_preprocessed['Education'].map(education_map).astype(float)

        # Handle Age_Group if present
        if 'Age_Group' in columns_to_encode and 'Age_Group' in df_preprocessed.columns:
            age_group_order = ['Young Adult', 'Middle Adult', 'Senior Adult']
            age_group_map = {group: idx for idx, group in enumerate(age_group_order)}
            df_preprocessed['Age_Group'] = df_preprocessed['Age_Group'].map(age_group_map).astype(float)

        # Handle Marital_Status if present
        if 'Marital_Status' in columns_to_encode and 'Marital_Status' in df_preprocessed.columns:
            # Create dummy variables with all categories
            marital_dummies = pd.get_dummies(
                df_preprocessed['Marital_Status'], 
                prefix='Marital_Status',
                drop_first=training_mode  # Only drop first category during training
            )
            
            # Drop original column and add encoded columns
            df_preprocessed = df_preprocessed.drop(columns=['Marital_Status'])
            df_preprocessed = pd.concat([df_preprocessed, marital_dummies], axis=1)

        return df_preprocessed
        
    except Exception as e:
        print(f"Error in feature encoding: {str(e)}")
        print(f"Available columns: {df_preprocessed.columns.tolist()}")
        print(f"Columns to encode: {columns_to_encode}")
        return df_preprocessed

## Feature scaling function
def feature_scaling(data, scalers=None, fit=True):
    """
    Scale features using appropriate scaling methods based on their distributions.
    """
    df_preprocessed = data.copy()
    
    # Define feature groups with their respective scaling methods
    feature_groups = {
        'log_transform': [
            'MntCoke', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
            'MntSweetProducts', 'MntGoldProds', 'Total_Spending', 'CVR'
        ],
        'count_based': [
            'NumWebVisitsMonth', 'NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases', 'Total_Purchases'
        ],
        'standard': [
            'Income', 'Age', 'Recency', 'Membership_Duration'
        ]
    }
    
    if scalers is None and fit:
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(quantile_range=(5, 95))
        }
    elif scalers is None and not fit:
        raise ValueError("Scalers must be provided when fit=False")

    try:
        # Process each feature group
        for group_name, features in feature_groups.items():
            # Get available features that exist in the dataframe
            available_features = [col for col in features if col in df_preprocessed.columns]
            
            if not available_features:
                continue
                
            # Convert to float
            df_preprocessed[available_features] = df_preprocessed[available_features].astype(float)
            
            # Apply log transformation for the log_transform group
            if group_name == 'log_transform':
                for feature in available_features:
                    df_preprocessed[feature] = np.log1p(df_preprocessed[feature])
                
                if fit:
                    df_preprocessed[available_features] = scalers['standard'].fit_transform(df_preprocessed[available_features])
                else:
                    df_preprocessed[available_features] = scalers['standard'].transform(df_preprocessed[available_features])
                    
            # Scale count-based features
            elif group_name == 'count_based':
                if fit:
                    df_preprocessed[available_features] = scalers['minmax'].fit_transform(df_preprocessed[available_features])
                else:
                    df_preprocessed[available_features] = scalers['minmax'].transform(df_preprocessed[available_features])
                    
            # Scale normally distributed features
            elif group_name == 'standard':
                if fit:
                    df_preprocessed[available_features] = scalers['standard'].fit_transform(df_preprocessed[available_features])
                else:
                    df_preprocessed[available_features] = scalers['standard'].transform(df_preprocessed[available_features])

        return df_preprocessed, scalers
        
    except Exception as e:
        print("Feature scaling error details:")
        print(f"Available columns: {df_preprocessed.columns.tolist()}")
        print(f"Attempted to scale: {[f for group in feature_groups.values() for f in group]}")
        raise Exception(f"Scaling error: {str(e)}\n\nAvailable columns: {df_preprocessed.columns.tolist()}")