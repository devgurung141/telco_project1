# imports
import pandas as pd
import numpy as np
import os
from env import get_connection
import acquire


# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_telco(df):
    '''
    This function takes in a dataframe and makes changes in a dataframe and return clean dataframe
    '''
    # drop unnecessary columns    
    df.drop(columns = ['payment_type_id', 'contract_type_id', 'internet_service_type_id' ] , inplace=True)
    
    # replace empty filed in total_charges columns and convert to float
    df.total_charges= df.total_charges.str.replace(' ','0').astype('float64')
    
    # creae a list of categorical variables
    cat_cols_1 = ['gender','partner','dependents','phone_service', 'paperless_billing', 'churn']
    cat_cols_2= ['multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','internet_service_type','contract_type','payment_type']
    
    # create dummy variable 
    dummies_1 = pd.get_dummies(df[cat_cols_1],drop_first=True)
    dummies_2 = pd.get_dummies(df[cat_cols_2])
    
    # concate dummy varibles with telco
    df = pd.concat([df, dummies_1, dummies_2],axis=1)
    
    #convert column names to lower case
    df.columns= df.columns.str.lower()
    
    return df


def train_validate_test_split(df, target, seed=42):
    '''
    This function takes in a dataframe and return train, validate and test dataframe; stratify on target
    '''
    
    # split data into 80% train_validate, 20% test
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    
    # split train_validate data into 70% train, 30% validate
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    # return train, validate, test
    return train, validate, test