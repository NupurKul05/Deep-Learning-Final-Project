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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

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

#%% Dimensionality reduction

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

VALID_AND_TEST_SIZE=0.1
rom sklearn.model_selection import train_test_split

X_train, X_else, y_train, y_else = train_test_split(df3, df3["pm2.5"], test_size=VALID_AND_TEST_SIZE*2, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_else, y_else, test_size=0.5, shuffle=False)