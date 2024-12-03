def get_feature_definitions():
    return {
        "ID": {
            "description": "Unique identifier for each customer",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Year_Birth": {
            "description": "Customer's year of birth",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "Dt_Customer": {
            "description": "Date of customer's registration with the company (joining date as a member)",
            "data_type": "Datetime",
            "specific_type": "-"
        },
        "Education": {
            "description": "Customer's level of education",
            "data_type": "Categorical",
            "specific_type": "Ordinal"
        },
        "Marital_Status": {
            "description": "Customer's marital status",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Kidhome": {
            "description": "Number of small children in the customer's household",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "Teenhome": {
            "description": "Number of teenagers in the customer's household",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "Income": {
            "description": "Customer's household income per year",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntFishProducts": {
            "description": "Amount spent on fish products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntMeatProducts": {
            "description": "Amount spent on meat products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntFruits": {
            "description": "Amount spent on fruit products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntSweetProducts": {
            "description": "Amount spent on sweet products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntCoke": {
            "description": "Amount spent on coke products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "MntGoldProds": {
            "description": "Amount spent on gold products in the last 2 years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "NumDealsPurchases": {
            "description": "Number of purchases made with a discount",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "NumCatalogPurchases": {
            "description": "Number of purchases made using a catalog (buying items to be shipped via mail)",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "NumStorePurchases": {
            "description": "Number of purchases made directly in stores",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "NumWebPurchases": {
            "description": "Number of purchases made through the company's website",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "AcceptedCmp1": {
            "description": "Whether the customer accept the first campaign or not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "AcceptedCmp2": {
            "description": "Whether the customer accept the second campaign or not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "AcceptedCmp3": {
            "description": "Whether the customer accept the third campaign or not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "AcceptedCmp4": {
            "description": "Whether the customer accept the fourth campaign or not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "AcceptedCmp5": {
            "description": "Whether the customer accept the fifth campaign or not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Complain": {
            "description": "1 if the customer complained in the last 2 years",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Recency": {
            "description": "Number of days since the customer's last purchase",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "Response": {
            "description": "1 if the customer responded to the offer in the last campaign, 0 if not",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        }
    } 