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

print("Content dataframe shape :", df2.shape)
print("Content dataframe columns :",df2.columns)

#%% Display the two original dataframes and the merged dataframe

df3 = pd.merge(df1, df2, left_on="product_sid", right_on="ProductId")
df3.head()
print(df3.shape)
pd.set_option('display.max_rows', 1000)
print(df3.isna().sum())