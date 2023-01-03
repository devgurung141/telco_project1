# imports

import pandas as pd
import numpy as np
import os
from env import get_connection

    
querry_telco = """
    SELECT * 
    FROM customers 
    LEFT JOIN internet_service_types USING(internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id);
    """


def get_telco_data():
    '''
    This function reads in telco data from Codeup database using sql querry into a df, 
    writes data to a csv file if a local file doesn't exist, and return df and if a local file exists
    return return df
    '''
    
    filename = "telco.csv"

    if os.path.isfile(filename):
        df =  pd.read_csv(filename)
        return df
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql(querry_telco, get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 