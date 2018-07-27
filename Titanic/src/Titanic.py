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
from sklearn import preprocessing

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
    
    # Age/Survived
    fig, ax = plt.subplots(1, 2, figsize = (18, 8))
    sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
    ax[0].set_title('Pclass and Age vs Survived')
    ax[0].set_yticks(range(0, 110, 10))
    
    sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
    ax[1].set_title('Sex and Age vs Survived')
    ax[1].set_yticks(range(0, 110, 10))
    
    # Age
    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    DataSet['Age'].hist(bins=70)
    plt.xlabel('Age')
    plt.ylabel('Num')

    plt.subplot(122)
    DataSet.boxplot(column='Age', showfliers=False)
    plt.show()

    facet = sns.FacetGrid(DataSet, hue = "Survived", aspect = 4)
    facet.map(sns.kdeplot, 'Age', shade = True)
    facet.set(xlim = (0, DataSet['Age'].max()))
    facet.add_legend()
    
    # average survived passsengers by age
    fig, axis1 = plt.subplots(1, 1, figsize = (18, 4))
    DataSet["Age_int"] = DataSet["Age"].astype(int)
    average_age = DataSet[["Age_int", "Survived"]].groupby(['Age_int'], as_index = False).mean()
    sns.barplot(x = 'Age_int', y = 'Survived', data = average_age)
    
    DataSet['Age'].describe()
    bins = [0, 12, 18, 65, 100]
    DataSet['Age_group'] = pd.cut(DataSet['Age'], bins)
    by_age = DataSet.groupby('Age_group')['Survived'].mean()
    by_age.plot(kind = 'bar')
    
    # name/Survived
    DataSet['Title'] = DataSet['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    pd.crosstab(DataSet['Title'], DataSet['Sex'])
    DataSet[['Title','Survived']].groupby(['Title']).mean().plot.bar()
    
    # namelength/Survived
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    DataSet['Name_length'] = DataSet['Name'].apply(len)
    name_length = DataSet[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
    sns.barplot(x='Name_length', y='Survived', data=name_length)

    # SibSp/Survived
    sibsp_df = DataSet[DataSet['SibSp'] != 0]
    no_sibsp_df = DataSet[DataSet['SibSp'] == 0]
    sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('sibsp')
    
    plt.subplot(122)
    no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('no_sibsp')
    plt.show()
    
    # Parch/Survived
    parch_df = DataSet[DataSet['Parch'] != 0]
    no_parch_df = DataSet[DataSet['Parch'] == 0]
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('parch')

    plt.subplot(122)
    no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('no_parch')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize = (18, 8))
    DataSet[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Parch and Survived')
    DataSet[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
    ax[1].set_title('SibSp and Survived')
    
    DataSet['Family_size'] = DataSet['Parch'] + DataSet['SibSp'] + 1
    DataSet[['Family_size','Survived']].groupby(['Family_size']).mean().plot.bar()
    
    # Fare/Survived
    plt.figure(figsize = (10, 5))
    DataSet['Fare'].hist(bins = 70)
    
    DataSet.boxplot(column = 'Fare', by = 'Pclass', showfliers = False)
    plt.show()
    
    DataSet['Fare'].describe()
    
    fare_not_survived = DataSet['Fare'][DataSet['Survived'] == 0]
    fare_survived = DataSet['Fare'][DataSet['Survived'] == 1]
    
    average_fare =  pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare  = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
    average_fare.plot(yerr=std_fare, kind='bar', legend=False)   
    plt.show()
    
    # Cabin/Survived
    DataSet.loc[DataSet.Cabin.isnull(), 'Cabin'] = 'U0'
    DataSet['Has_Cabin'] = DataSet['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
    DataSet[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()
    # create feature for the alphabetical part of the cabin number
    DataSet['CabinLetter'] = DataSet['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    # convert the distinct cabin letters with incremental integer values
    DataSet['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
    DataSet[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()
    
    # Embarked/Survived
    sns.countplot('Embarked', hue='Survived', data = DataSet)
    plt.title('Embarked and Survived')
    
    sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
    plt.title('Embarked and Survived rate')
    plt.show()
    
def qualitativeTransfer(DataSet):
    
    embark_dummies = pd.get_dummies(DataSet['Embarked'])
    DataSet = DataSet.join(embark_dummies)
    DataSet.drop(['Embarked'], axis = 1, inplace = True)
    embark_dummies.head()
    
    # Replace missing values with "U0"
    DataSet['Cabin'][DataSet.Cabin.isnull()] = 'U0'
    # create feature for the alphabetical part of the cabin number
    DataSet['CabinLetter'] = DataSet['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    # convert the distinct cabin letters with incremental integer values
    DataSet['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
    DataSet['CabinLetter'].head()

def quantitativeTransfer(DataSet):
    
    assert np.size(DataSet['Age']) == 891
    # StandardScaler will subtract the mean from each value then scale to the unit variance
    scaler = preprocessing.StandardScaler()
    DataSet['Age_scaled'] = scaler.fit_transform(DataSet['Age'].values.reshape(-1, 1))
    
    DataSet['Age_scaled'].head()
    
    # Divide all fares into quartiles
    DataSet['Fare_bin'] = pd.qcut(DataSet['Fare'], 5)
    DataSet['Fare_bin'].head()
    
    # qcut() creates a new variable that identifies the quartile range, but we can't use the string
    # so either factorize or create dummies from the result
    # factorize
    DataSet['Fare_bin_id'] = pd.factorize(DataSet['Fare_bin'])[0]
    
    # dummies
    fare_bin_dummies_df = pd.get_dummies(DataSet['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
    DataSet = pd.concat([DataSet, fare_bin_dummies_df], axis=1)
    

def featureEnginner(DataSet):
    
    # use most Embarked to fillna
    DataSet['Embarked'].fillna(DataSet['Embarked'].mode().iloc[0], inplace=True)
    DataSet['Embarked'] = pd.factorize(DataSet['Embarked'])[0]
    
    emb_dummies_df = pd.get_dummies(DataSet['Embarked'], prefix=DataSet[['Embarked']].columns[0])
    DataSet = pd.concat([DataSet, emb_dummies_df], axis=1)

    # Sex
    DataSet['Sex'] = pd.factorize(DataSet['Sex'])[0]
    
    sex_dummies_df = pd.get_dummies(DataSet['Sex'], prefix=DataSet[['Sex']].columns[0])
    DataSet = pd.concat([DataSet, sex_dummies_df], axis=1)

    # Name
    # what is each person's title? 
    DataSet['Title'] = DataSet['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])    
    # unit the Title
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    DataSet['Title'] = DataSet['Title'].map(title_Dict)
    # factorize and dummy
    DataSet['Title'] = pd.factorize(DataSet['Title'])[0]
    title_dummies_df = pd.get_dummies(DataSet['Title'], prefix=DataSet[['Title']].columns[0])
    DataSet = pd.concat([DataSet, title_dummies_df], axis=1)

    # add name len feature
    DataSet['Name_length'] = DataSet['Name'].apply(len)

# main test
if __name__ == '__main__':
    
    #train_data, test_data = loadData()
    
    #train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
    
    #nanValueDeal(train_data)
    
    #dataAnyls(train_data)
    
    #qualitativeTransfer(train_data)
    
    #quantitativeTransfer(train_data)
    
    # need to do feature enginner
    train_df_org = pd.read_csv('../data/train.csv')
    test_df_org = pd.read_csv('../data/test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = train_df_org.append(test_df_org)
    PassengerId = test_df_org['PassengerId']
    
    featureEnginner(combined_train_test)
    
