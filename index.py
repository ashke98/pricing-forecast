import pandas as pd
import csv

from helpers import *

# last 12 months are part of the test set
# rest of the data is used to train
# using a min-max scaler, we will scale the data so that all of our variables fall within the range of -1 to 1
# Reverse scaling: After running our models, we will use this helper function to reverse the scaling
# we will save the root mean squared error (RMSE) and mean absolute error (MAE) of our predictions to compare performance of our five models
def load_data(csv_path):
    return pd.read_csv(csv_path)

def monthly_sales(data):    
    data = data.copy()     
    # Drop the day indicator from the date column    
    data.date = data.date.apply(lambda x: str(x)[:-3])     
    # Sum sales per month    
    data = data.groupby('date')['sales'].sum().reset_index()    
    data.date = pd.to_datetime(data.date)  
    data.to_csv('data/monthly_data.csv')     
    return data


# Calculate difference in sales month over month
def get_diff(data):
    data['sales_diff'] = data.sales.diff()    
    data = data.dropna()      
    return data

# ARIMA Model
def generate_arima_data(data):
    dt_data = data.set_index('date').drop('sales', axis=1)        
    dt_data.dropna(axis=0)     
    dt_data.to_csv('data/arima_df.csv')
    return dt_data

# Supervised Model
def generate_supervised(data):
    supervised_df = data.copy()
    
    #create column for each lag
    for i in range(1,13):
        col = 'lag_' + str(i)
        supervised_df[col] = supervised_df['sales_diff'].shift(i)
    
    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    supervised_df.to_csv('data/model_df.csv', index=False)
    
    return supervised_df

data = load_data("data/train.csv")
monthly_data = monthly_sales(data)

# means keep data to no be change-able
stationary_df = get_diff(monthly_data)

# set data for arima model -> datetime index
print("Generating Arima Model")
arima_data = generate_arima_data(stationary_df)

# set data for supervised model -> lags as features
print("Generating Supervised Model")
model_df = generate_supervised(stationary_df)

#################################################################################### MODELING
# Regressive Models: Linear Regression, Random Forest Regression, XGBoost

def regressive_model(train_data, test_data, model, model_name):
    
    # Call helper functions to create X & y and scale data
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    # Run regression model
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    # Call helper functions to undo scaling & create prediction df
    original_df = load_data('data/monthly_data.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)
    # Call helper functions to print scores and plot results
    get_scores(unscaled_df, original_df, model_name)
    plot_results(unscaled_df, original_df, model_name)

# Separate data into train and test sets
print("train, test")
train, test = tts(model_df)

# Call model frame work for linear regression
print("regressive_model -> LinearRegression")
regressive_model(train, test, LinearRegression(),'LinearRegression')

# Call model frame work for random forest regressor 
print("regressive_model -> RandomForest")
regressive_model(train, test, 
                 RandomForestRegressor(n_estimators=100,
                                       max_depth=20),        
                                       'RandomForest')
# Call model frame work for XGBoost
print("regressive_model -> XGBoost")
regressive_model(train, test, XGBRegressor(n_estimators=100,
                                           learning_rate=0.2), 
                                           'XGBoost')



###############################################################

# Long Short-Term Memory (LSTM)
# For additional accuracy, seasonal features and additional model complexity can be added
def lstm_model(train_data, test_data):
    # Call helper functions to create X & y and scale data
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    # Build LSTM
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], 
                                         X_train.shape[2]), 
                                         stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, 
              shuffle=False)
    predictions = model.predict(X_test, batch_size=1)
    # Call helper functions to undo scaling & create prediction df
    original_df = load_data('data/monthly_data.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object, 
                            lstm=True)
    unscaled_df = predict_df(unscaled, original_df)
    # Call helper functions to print scores and plot results
    get_scores(unscaled_df, original_df, 'LSTM')
    plot_results(unscaled_df, original_df, 'LSTM')


def sarimax_model(data):
    # Model    
    sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(12, 0, 
                                    0), seasonal_order=(0, 1, 0,  
                                    12), trend='c').fit()
    # Generate predictions    
    start, end, dynamic = 40, 100, 7    
    data['pred_value'] = sar.predict(start=start, end=end, 
                                     dynamic=dynamic)     
    # Call helper functions to undo scaling & create prediction df   
    original_df = load_data('data/monthly_data.csv')
    unscaled_df = predict_df(data, original_df)
    # Call helper functions to print scores and plot results   
    get_scores(unscaled_df, original_df, 'ARIMA') 
    plot_results(unscaled_df, original_df, 'ARIMA')

def create_results_df():
    # Load pickled scores for each model
    results_dict = pickle.load(open("model_scores.p", "rb"))
    # Create pandas df 
    results_df = pd.DataFrame.from_dict(results_dict, 
                    orient='index', columns=['RMSE', 'MAE', 'R2'])
    results_df = results_df.sort_values(by='RMSE',
                     ascending=False).reset_index()
    return results_df


results = create_results_df()