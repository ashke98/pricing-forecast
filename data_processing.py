"""
This script loads data from Kaggle, generates monthly dataframe and performs
differencing to create stationarity. Exports csv files for regression
modeling and for Arima modeling.
Data: https://www.kaggle.com/c/demand-forecasting-kernels-only/data
Output CSV files
 -- data/wallmart/monthly_data.csv
 -- data/wallmart/stationary_df.csv
 -- data/wallmart/model_df.csv
 -- data/wallmart/arima_df.csv
"""

# url = """https://www.kaggle.com/c/demand-forecasting-kernels-only/download/
#          ryQFx3IEtFjqjv3s0dXL%2Fversions%2FzjbSfpE39fdJlMotCpen%2Ffiles%2
#          Ftrain.csv"""

import pandas as pd

def converter(path):
    walmart_data = pd.read_csv(path)

    data = walmart_data.copy()
    data = data.drop(['SHOP_WEEKDAY','SHOP_WEEKDAY','SHOP_HOUR' ,'SPEND','CUST_LIFESTAGE','BASKET_ID','BASKET_SIZE','BASKET_PRICE_SENSITIVITY','BASKET_DOMINANT_MISSION','STORE_FORMAT','STORE_REGION'], axis=1)     
    data.to_csv('data/wallmart/train.csv')

def load_data(path):
    return pd.read_csv(path)

def daily_sales(data):
    daily_data = data.copy()

    daily_data.date = daily_data.date.apply(lambda x: str(x)[0:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8])  
       
    # Sum sales per month    
    daily_data = daily_data.groupby('date')['sales'].sum().reset_index()    
    daily_data.date = pd.to_datetime(daily_data.date)  
    daily_data.to_csv('data/wallmart/daily_data.csv')

    return daily_data

def monthly_sales(data):
    """Returns a dataframe where each row represents total sales for a given
    month. Columns include 'date' by month and 'sales'.
    """
    monthly_data = data.copy()

    # Drop the day indicator from the date column
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])

    # Sum sales per month
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)

    monthly_data.to_csv('data/wallmart/monthly_data.csv')

    return monthly_data

def get_diff(data):
    """Returns the dataframe with a column for sales difference between each
    month. Results in a stationary time series dataframe. Prior EDA revealed
    that the monthly data was not stationary as it had a time-dependent mean.
    """
    data['sales_diff'] = data.sales.diff()
    data = data.dropna()

    data.to_csv('data/wallmart/stationary_df.csv')

    return data

def generate_supervised(data):
    """Generates a csv file where each row represents a month and columns
    include sales, the dependent variable, and prior sales for each lag. Based
    on EDA, 12 lag features are generated. Data is used for regression modeling.
    Output df:
    month1  sales  lag1  lag2  lag3 ... lag11 lag12
    month2  sales  lag1  lag2  lag3 ... lag11 lag12
    """
    supervised_df = data.copy()

    #create column for each lag
    for i in range(1, 13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)

    supervised_df.to_csv('data/wallmart/model_df.csv', index=False)

def generate_arima_data(data):
    """Generates a csv file with a datetime index and a dependent sales column
    for ARIMA modeling.
    """
    dt_data = data.set_index('date').drop('sales', axis=1)
    dt_data.dropna(axis=0)

    dt_data.to_csv('data/wallmart/arima_df.csv')


def main():
    """Loads data from Kaggle, generates monthly dataframe and performs
    differencing to create stationarity. Exports csv files for regression
    modeling and for Arima modeling.
    """
    converter('data/trainWallmart.csv')
    sales_data = load_data('data/wallmart/train.csv')
    daily_df = daily_sales(sales_data)
    stationary_df = get_diff(daily_df)

    generate_supervised(stationary_df)
    generate_arima_data(stationary_df)

    # monthly_df = monthly_sales(sales_data)
    # stationary_df = get_diff(monthly_df)

    # generate_supervised(stationary_df)
    # generate_arima_data(stationary_df)

main()