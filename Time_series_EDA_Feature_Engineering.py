# -*- coding: utf-8 -*-
"""
Created on 13th May 2019

@author: Ananya
"""

#%% [Markdown]
#%% In Jupyter notebook this representation acts as a headee

#%% Importing libraries

import os
import numpy as np
import pandas as pd
import sklearn as sk
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% Reading the datasets as two different dataframes

os.chdir("/Users/ananyaneogi/Desktop/FS-SECOND SEM/Deep learning Project/Homework_Assignment")
df1 = pd.read_csv('solditems_encoded_stage2.csv')
df2 = pd.read_csv('content_encoded_stage2.csv')

#%% How does the data look?

print("Solid Item dataframe shape :", df1.shape)
print("Solid Item dataframe columns :",df1.columns)
df1.head(5)

#A copy of the existing dataframe made for alteration purposes
df1_new = df1.copy()

# Type of the date id changed to datetime format
df1_new['created_date_id'] = pd.to_datetime(df1_new['created_date_id'])

# From the date id the date field is extracted for further grouping purposes
df1_new['year'] = df1_new['created_date_id'].dt.year
df1_new.head(5)
df1_new.groupby('product_sid').apply(lambda x: x.sort_values(['year'], ascending = True))

# Total units sold of a product for every year.
product_count = df1_new.groupby(['year', 'product_sid']).size()

print("Content dataframe shape :", df2.shape)
print("Content dataframe columns :",df2.columns)
df2.describe()
pd.set_option('display.max_rows', 1000)

manufacturers_to_drop = ['Manufacturer_1','Manufacturer_2','Manufacturer_9','Manufacturer_4','Manufacturer_5',
                        'Manufacturer_6','Manufacturer_7','Manufacturer_8','Manufacturer_10','Manufacturer_11',
                        'Manufacturer_12','Manufacturer_13','Manufacturer_15','Manufacturer_16','Manufacturer_17',
                        'Manufacturer_18','Manufacturer_19','Manufacturer_20','Manufacturer_21','Manufacturer_22',
                        'Manufacturer_23','Manufacturer_24','Manufacturer_25','Manufacturer_26','Manufacturer_27',
                        'Manufacturer_28','Manufacturer_29','Manufacturer_30','Manufacturer_31','Manufacturer_32',
                        'Manufacturer_34','Manufacturer_35','Manufacturer_36','Manufacturer_37','Manufacturer_38',
                        'Manufacturer_39','Manufacturer_40','Manufacturer_41','Manufacturer_42','Manufacturer_43',
                        'Manufacturer_44','Manufacturer_45','Manufacturer_46','Manufacturer_47','Manufacturer_48',
                        'Manufacturer_49','Manufacturer_50','Manufacturer_51','Manufacturer_52','Manufacturer_53',
                        'Manufacturer_54','Manufacturer_55','Manufacturer_56','Manufacturer_57','Manufacturer_58',
                        'Manufacturer_59','Manufacturer_60','Manufacturer_61','Manufacturer_62','Manufacturer_63',
                        'Manufacturer_64','Manufacturer_65','Manufacturer_66','Manufacturer_67','Manufacturer_68',
                        'Manufacturer_69','Manufacturer_70','Manufacturer_71','Manufacturer_72','Manufacturer_73',
                        'Manufacturer_74']
df2 = df2.drop(manufacturers_to_drop, axis = 1)
df2.shape

count_rows = df2.apply(lambda x: True if x['Manufacturer_0'] or x['Manufacturer_3'] or x['Manufacturer_14'] or x['Manufacturer_33'] == 1 else False , axis=1)
num_rows = len(count_rows[count_rows == True].index)
print('Number of Rows where value is 1 for the manufacturers are: ', num_rows)

df2 = df2[df2['Manufacturer_0'] | df2['Manufacturer_3'] | df2['Manufacturer_14'] | df2['Manufacturer_33'] == 1] 
df2.shape


#%% Display the two original dataframes and the merged dataframe

df3 = pd.merge(df1, df2, left_on="product_sid", right_on="ProductId")
print(df3.head())
print(df3.shape)
pd.set_option('display.max_rows', 1000)
print(df3.isna().sum())

print(df3.describe())

#%% Partial Autocorrelation Plot

from statsmodels.graphics.tsaplots import plot_pacf

#columns = [] #use this for speedup
columns = ["sales_item_price", 'sales_voucher_created', 'sales_voucher',
       'sales_value_created', 'sales_value','created_date_id', 'sales_item_price_created',
       'days_since_first_sold', 'days_since_release', 'returned_date_id_0',
       'returned_date_id_1']

for col in columns:
    plt.figure()
    plot_pacf(df1_new[col].dropna(), lags=48, zero=False)
    plt.title("Partial Autocorrelation PLot : " + str(col))
    
plt.show()

#%% 

from statsmodels.graphics.tsaplots import _prepare_data_corr_plot, _plot_corr
import statsmodels.graphics.utils as utils
from statsmodels.tsa.stattools import pacf

def plot_pacf_drop(x, ax=None, lags=None, alpha=.05, method='ywunbiased',
              use_vlines=True, title='Partial Autocorrelation', zero=True,
              vlines_kwargs=None, drop_no=0, **kwargs):
    
    lags_orig=lags
    fig, ax = utils.create_mpl_ax(ax)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)
    confint = None
    if alpha is None:
        acf_x = pacf(x, nlags=nlags, alpha=alpha, method=method)
    else:
        acf_x, confint = pacf(x, nlags=nlags, alpha=alpha, method=method)

    if drop_no:
        acf_x = acf_x[drop_no+1:]
        confint = confint[drop_no+1:]
        lags, nlags, irregular = _prepare_data_corr_plot(x, lags_orig-drop_no, zero)
        
    _plot_corr(ax, title, acf_x, confint, lags, False, use_vlines,
               vlines_kwargs, **kwargs)

    return fig

    import matplotlib.pyplot as plt

#columns = [] #use this for speedup
columns = ["sales_item_price", 'sales_voucher_created', 'sales_voucher',
       'sales_value_created', 'sales_value','created_date_id', 'sales_item_price_created',
       'days_since_first_sold', 'days_since_release', 'returned_date_id_0',
       'returned_date_id_1']

for col in columns:

    plt.figure()
    plot_pacf_drop(df1_new[col].dropna(), lags=200, drop_no=3, zero=False)
    plt.title("Partial Autocorrelation PLot : " + str(col))
    
plt.show()

#%%

#grouped = df1.groupby(['product_sid'])
#l_grouped = list(grouped)
#l_grouped[0][1]

#l_grouped.product_sid.unique()


#%% Split the data into training and test sets

from seglearn.split import temporal_split

X_train, X_else, y_train, y_else = train_test_split(df1_new, df1_new["sales_item_price_created"], test_size=0.2, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_else, y_else, test_size=0.5, shuffle=False)

#X_train, X_valid, y_train, y_valid = temporal_split(df1_new, df1_new["sales_item_price_created"], test_size=0.25)

#normalizers = minmax_scale(X_train, y_train)
#%% 

TIME_WINDOW=100
FORECAST_DISTANCE=24

from seglearn.transform import FeatureRep, SegmentXYForecast, last

segmenter = SegmentXYForecast(width=TIME_WINDOW, step=1, y_func=last, forecast=FORECAST_DISTANCE)

X_train_rolled, y_train_rolled,_=segmenter.fit_transform([X_train.values],[y_train])

X_train_rolled

#%%

X_train_rolled.shape

shape = X_train_rolled.shape
X_train_flattened = X_train_rolled.reshape(shape[0],shape[1]*shape[2])
X_train_flattened.shape


X_valid_rolled, y_valid_rolled,_=segmenter.fit_transform([X_valid.values],[y_valid])

shape = X_valid_rolled.shape
X_valid_flattened = X_valid_rolled.reshape(shape[0],shape[1]*shape[2])

X_valid_flattened
#%%

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning

def evaluate_model(model, X_valid, y_valid_true):
    predictions = model.predict(X_valid)
    rms = sqrt(mean_squared_error(y_valid_true, predictions))
    print("Root mean squared error on valid:",rms)
    #normalized_rms = df1_new["sales_item_price_created"].inverse_transform(np.array([rms]).reshape(1, -1))[0][0]
    #print("Root mean squared error on valid inverse transformed from normalization:",normalized_rms)
    return rms

#%% Dummy Regressor

from sklearn.dummy import DummyRegressor

dummy_model = DummyRegressor(strategy="mean", constant=None, quantile=None)

dummy_model.fit(X_train_flattened,y_train_rolled)

result = evaluate_model(dummy_model,X_valid_flattened,y_valid_rolled)

#%% XGBoost

import xgboost as xgb
# If in trouble, use !pip install xgboost

# XGBoost needs it's custom data format to run quickly
dmatrix_train = xgb.DMatrix(data=X_train_flattened,label=y_train_rolled)
dmatrix_valid = xgb.DMatrix(data=X_valid_flattened,label=y_valid_rolled)

params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'n_estimators': 20}

evallist = [(dmatrix_valid, 'eval'), (dmatrix_train, 'train')]

num_round = 10 #Can easily overfit, experiment with it!

xg_reg = xgb.train(params, dmatrix_train, num_round,evallist)

result = evaluate_model(xg_reg,dmatrix_valid,y_valid_rolled)

#%% LSTM

LSTM_CELL_SIZE=350
BATCH_SIZE = 300
EPOCHS = 60
DROPOUT_RATE=0

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from tensorflow.keras import backend as be
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

column_count=len(X_train.columns) #Remember,column count before rolling...

be.clear_session()

# You might very well be needing it!
# Remeber to save only what is worth it from validation perspective...
# model_saver = ModelCheckpoint(...)

# If you need it...
#def schedule(epoch, lr):
#    ...
#    return lr

#lr_scheduler = LearningRateScheduler(schedule)

# Build your whole LSTM model here!
model = Sequential()

model.add(CuDNNLSTM(LSTM_CELL_SIZE, input_shape=(TIME_WINDOW,column_count),stateful=False))
model.add(Dense(1, activation= "linear"))


#For shape remeber, we have a variable defining the "window" and the features in the window...

model.compile(loss='mean_squared_error', optimizer='sgd')
# Fit on the train data
# USE the batch size parameter!
# Use validation data - warning, a tuple of stuff!
# Epochs as deemed necessary...
# You should avoid shuffling the data maybe.
# You can use the callbacks for LR schedule or model saving as seems fit.
history = model.fit(X_train_rolled, y_train_rolled, batch_size=BATCH_SIZE, epochs=EPOCHS,
          validation_data=(X_valid_rolled ,y_valid_rolled), shuffle=False)

# Plot the loss function of training and test sets
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

result = evaluate_model(model,X_valid_rolled ,y_valid_rolled)