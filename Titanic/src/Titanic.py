#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: Titanic.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-25
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.ensemble import RandomForestRegressor 

warnings.filterwarnings('ignore')

def loadData():
    
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    sns.set_style('whitegrid')
    train_data.head()

    train_data.info()
    print("-" * 40)
    test_data.info()
    
    return train_data, test_data

def nanValueDeal(DataSet):
    
    # Embarked feature use vote to deal
    DataSet.Embarked[train_data.Embarked.isnull()] = DataSet.Embarked.dropna().mode().values
    
    # Cabin feature use U0 to deal
    DataSet['Cabin'] = DataSet.Cabin.fillna('U0')
    
    # use random forest to predict age
    age_df = DataSet[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(DataSet['Age'].notnull())]
    age_df_isnull = age_df.loc[(DataSet['Age'].isnull())]
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]
    
    # use RandomForestRegression to train data
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X,Y)
    
    predictAges = RFR.predict(age_df_isnull.values[:,1:])
    DataSet.loc[DataSet['Age'].isnull(), ['Age']] = predictAges
    
    DataSet.info()

def dataAnyls(DataSet):
    
    # sex/survived 
    DataSet[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
    
    # Pclass/Survived
    DataSet[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()


# main test
if __name__ == '__main__':
    
    train_data, test_data = loadData()
    
    train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
    
    nanValueDeal(train_data)
    
    dataAnyls(train_data)
    
    
    