# Created By: Gregory P. Winter
# email: gregorypwinter@gmail.com

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def impute_reg(col1, df, missing=np.nan):
    if missing != np.nan:
        df[col1].replace(to_replace=missing, value=np.nan, inplace=True)
        
    # check column type, if wrong type kick back message
    if str(df[col1].dtype) not in ('float32', 'float64'):
        print "Wrong Type"
        return
    
    column_names = list(df.columns)
    column_names.remove(col1) # remove y from dataframe
    nex = pd.get_dummies(df[column_names], drop_first = True) # Create dummy columns
    target = df[col1] # set y value
    nex = pd.concat([target, nex], axis = 1) # Concat y with x
    train = nex[pd.isnull(nex[col1]) == False] # filter for missing values
    pred = nex[pd.isnull(nex[col1]) == True]
    if len(pred) == 0: # Check to see if anything is missing if not kick back message.
        print "No Missing / NaN Data or no missing value designated."
        return
    
    y = train[col1] # create y value
    train.drop(col1, axis = 1, inplace = True) # remove y from train set
    pred.drop(col1, axis = 1, inplace = True) # remove y (missing info) from prediction set
    XGB = XGBRegressor(n_jobs=-1)
    XGB.fit(train, y) # import and train XGB
    for x in list(pred.index): # get predictions and place them in orginal dataframe
        value = XGB.predict(pred.loc[[x]])
        df.loc[x, col1] = value[0]
        
    return df

def impute_cal(col1, df, missing=np.nan):
    if missing != np.nan:
        # Change specified values to np.nan if instructed.
        df[col1].replace(to_replace=missing, value=np.nan, inplace=True)
        
    # check column type, if wrong type kick back message.
    if str(df[col1].dtype) not in ['object', 'category']:
        print "Wrong Type"
        return
    
    target = df[col1]
    column_names = list(df.columns)
    column_names.remove(col1)
    nex = pd.get_dummies(df, drop_first = True)
    nex = pd.concat([target, nex], axis = 1)
    train = nex[pd.isnull(nex[col1]) == False]
    pred = nex[pd.isnull(nex[col1]) == True]
    if len(pred) == 0:
        print "No Missing / NaN Data or no missing value designated."
        return
    
    y = train[col1]
    pred.drop(col1, axis = 1, inplace = True)
    train.drop(col1, axis = 1, inplace = True)    
    XGB = XGBClassifier(n_jobs=-1)
    XGB.fit(train, y)
    for x in list(pred.index):
        value = XGB.predict(pred.loc[[x]])
        df.loc[x, col1] = value[0]
    return df

def dataset():
    data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, -1, 0, -1],
        'sprinkler': [0, 1, 1, 0, 1, 0, 1, -1],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, -1, 0],
        'some_num': [1.1, np.NaN, 0.2, -0.4, 0.1, 0.2, 0.0, 3.9],
        'some_str': ['B', 'A', 'A', 'A', 'A', 'A', 'A', np.NaN]
    })
    return data

def dataset1():
    data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, -1, 0, -1],
        'sprinkler': [0, 1, 1, 0, 1, 0, 1, -1],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, -1, 0],
        'some_num': [1.1, -10, 0.2, -0.4, 0.1, 0.2, 0.0, 3.9],
        'some_str': ['B', 'A', 'A', 'A', 'A', 'A', 'A', 'Cars']
    })
    return data
