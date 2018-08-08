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
from sklearn.preprocessing import LabelEncoder

import warnings

from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

def loadData():
    
    train_data = pd.read_csv('../input/train.csv')
    test_data = pd.read_csv('../input/test.csv')
    
    sns.set_style('whitegrid')
    train_data.head(5)

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

def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'    

def family_size_category(family_size):
    
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

def fill_missing_age(missing_age_train, missing_age_test):
  
    missing_age_X_train = missing_age_train.drop(['Age'], axis = 1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis = 1)
    
    # model 1 gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    
    print("Age feature Best GB Params:" + str(gbm_reg_grid.best_params_))
    print("Age feature Best GB Score:" + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    
    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])
    
    # two models merge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis = 1).shape)
    # missing_age_test['Age'] =  missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)
     
    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])
     
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
     
    return missing_age_test
 
    
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
    
    # Fare
    DataSet['Fare'] = DataSet[['Fare']].fillna(DataSet.groupby('Pclass').transform(np.mean))
        
    DataSet['Group_Ticket'] = DataSet['Fare'].groupby(by=DataSet['Ticket']).transform('count')
    DataSet['Fare'] = DataSet['Fare'] / DataSet['Group_Ticket']
    DataSet.drop(['Group_Ticket'], axis=1, inplace=True)
    # bining
    DataSet['Fare_bin'] = pd.qcut(DataSet['Fare'], 5)    
    # factorize and dummy
    DataSet['Fare_bin_id'] = pd.factorize(DataSet['Fare_bin'])[0]
    fare_bin_dummies_df = pd.get_dummies(DataSet['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
    DataSet = pd.concat([DataSet, fare_bin_dummies_df], axis=1)
    DataSet.drop(['Fare_bin'], axis=1, inplace=True)
    
    # Pclass
    Pclass1_mean_fare = DataSet['Fare'].groupby(by=DataSet['Pclass']).mean().get([1]).values[0]
    Pclass2_mean_fare = DataSet['Fare'].groupby(by=DataSet['Pclass']).mean().get([2]).values[0]
    Pclass3_mean_fare = DataSet['Fare'].groupby(by=DataSet['Pclass']).mean().get([3]).values[0]
    # build Pclass_Fare Category
    DataSet['Pclass_Fare_Category'] = DataSet.apply(pclass_fare_category, args=(
    Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
    pclass_level = LabelEncoder()
    
    pclass_level.fit(np.array(
    ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
    # dummy pclass category
    DataSet['Pclass_Fare_Category'] = pclass_level.transform(DataSet['Pclass_Fare_Category'])
    pclass_dummies_df = pd.get_dummies(DataSet['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
    DataSet = pd.concat([DataSet, pclass_dummies_df], axis=1)
    # factorize
    DataSet['Pclass'] = pd.factorize(DataSet['Pclass'])[0]
    
    # Parch and SibSp
    DataSet['Family_Size'] = DataSet['Parch'] + DataSet['SibSp'] + 1
    DataSet['Family_Size_Category'] = DataSet['Family_Size'].map(family_size_category)
    
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    DataSet['Family_Size_Category'] = le_family.transform(DataSet['Family_Size_Category'])
    
    family_size_dummies_df = pd.get_dummies(DataSet['Family_Size_Category'],
                                        prefix=DataSet[['Family_Size_Category']].columns[0])
    DataSet = pd.concat([DataSet, family_size_dummies_df], axis=1)

    # predict Age for nan
    missing_age_df = pd.DataFrame(DataSet[['Age', 'Embarked', 'Sex', 'Title', 
                                           'Name_length', 'Family_Size', 'Family_Size_Category',
                                           'Fare', 'Fare_bin_id', 'Pclass']])

    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
    missing_age_test.head()
    
    DataSet.loc[(DataSet.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
    
    # Ticket
    DataSet['Ticket_Letter'] = DataSet['Ticket'].str.split().str[0]
    DataSet['Ticket_Letter'] = DataSet['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
    
    # Ticket_Letter factorize
    DataSet['Ticket_Letter'] = pd.factorize(DataSet['Ticket_Letter'])[0]
    
    # Cabin
    DataSet.loc[DataSet.Cabin.isnull(), 'Cabin'] = 'U0'
    DataSet['Cabin'] = DataSet['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
        
    return DataSet

def featureRelationshipDisp(DataSet):
    
    Correlation = pd.DataFrame(DataSet[['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 
                                        'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 
                                        'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
    colormap = plt.cm.viridis
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)

    g = sns.pairplot(DataSet[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked', u'Family_Size', u'Title', u'Ticket_Letter']], 
                     hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
    g.set(xticklabels=[])

def preDataSet(DataSet):
    
    # regulariz Age/fare
    scale_age_fare = preprocessing.StandardScaler().fit(DataSet[['Age','Fare', 'Name_length']])
    DataSet[['Age','Fare', 'Name_length']] = scale_age_fare.transform(DataSet[['Age','Fare', 'Name_length']])

    # dropout feature
    combined_data_backup = DataSet
    DataSet.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 
                          'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)
    
    # split training and testing data
    train_data = DataSet[:891]
    test_data = DataSet[891:]

    titanic_train_data_X = train_data.drop(['Survived'],axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'],axis=1)
    titanic_train_data_X.shape
    
    return titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X


def getTopNFeature(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))
    
    # AdaBoost
    ada_est =AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))
    
    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))
    
    # merge the three models
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 
                               ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 
                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)

    return features_top_n , features_importance


def get_out_fold(clf, x_train, y_train, x_test, ntrain, ntest):

    SEED = 0 # for reproducibility
    NFOLDS = 7 # set folds for out-of-fold prediction
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def modelEnsemble(titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X):
    
    # L1
    # Some useful parameters which will come in handy later on
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    #SEED = 0 # for reproducibility
    #NFOLDS = 7 # set folds for out-of-fold prediction
    #kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)
    
    # Create 7 baseline learn model
    # randomForest
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    # Adaboost
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    # ExtraTrees
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    # GBDT
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
    # DecisionTree
    dt = DecisionTreeClassifier(max_depth=8)
    # KNN
    knn = KNeighborsClassifier(n_neighbors = 2)
    # SVM
    svm = SVC(kernel='linear', C=0.025)
    
    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    x_train = titanic_train_data_X.values 
    x_test = titanic_test_data_X.values 
    y_train = titanic_train_data_Y.values
   
    # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test, ntrain, ntest) # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test, ntrain, ntest) # AdaBoost 
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test, ntrain, ntest) # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test, ntrain, ntest) # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test, ntrain, ntest) # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test, ntrain, ntest) # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test, ntrain, ntest) # Support Vector
    
    print("Training is complete")
    
    # L2
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    gbm = XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('../output/SubmitResult.csv',index=False,sep=',')
        

# main test
if __name__ == '__main__':
    
    #train_data, test_data = loadData()
    
    #train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
    
    #nanValueDeal(train_data)
    
    #dataAnyls(train_data)
    
    #qualitativeTransfer(train_data)
    
    #quantitativeTransfer(train_data)
    
    # need to do feature enginner
    train_df_org = pd.read_csv('../input/train.csv')
    test_df_org = pd.read_csv('../input/test.csv')
    test_df_org['Survived'] = 0
    DataSet = train_df_org.append(test_df_org)
    PassengerId = test_df_org['PassengerId']

    DataSet = featureEnginner(DataSet)
    #featureRelationshipDisp(DataSet)
    
    titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X = preDataSet(DataSet)
    feature_to_pick = 30
    feature_top_n, feature_importance = getTopNFeature(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
    
    modelEnsemble(titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X)
    
    
    
    
