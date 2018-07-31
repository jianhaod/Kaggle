#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: HousePrices.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-31
"""

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loadDataSet():
    
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    DataSet = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']), 
                        ignore_index = True)
    DataSet.shape
    return DataSet, train, test

def correlationAnalys(trainData):
    
    # traindata correlation
    corrmat = trainData.corr()
    plt.subplots(figsize = (12, 9))
    sns.heatmap(corrmat, vmax = 0.9, square = True)
    
    k = 10
    plt.figure(figsize = (12, 9))
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(trainData[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar = True, square = True, fmt='.2f', 
                     annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
    plt.show()
    
    Corr = trainData.corr()
    Corr[Corr['SalePrice'] > 0.5]
    
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(trainData[cols], size = 2.5)
    plt.show()

def distributionAnalys(trainData):
    
    trainData['SalePrice'].describe()
    sns.distplot(trainData['SalePrice'], fit = stats.norm)
    fig = plt.figure()
    res = stats.probplot(trainData['SalePrice'], plot = plt)
    
    print("Skewness: %f" %trainData['SalePrice'].skew())
    print("Kurtosis: %f" %trainData['SalePrice'].kurt())
    
    # SalePrice and GrLiveArea relationship
    data1 = pd.concat([trainData['SalePrice'], trainData['GrLivArea']], axis=1)
    data1.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
    
    fig = plt.figure()
    sns.distplot(trainData['GrLivArea'], fit = stats.norm)
    fig = plt.figure()
    res = stats.probplot(trainData['GrLivArea'], plot=plt)

    # SalePrice and TotalBsmtSF
    data1 = pd.concat([trainData['SalePrice'], trainData['TotalBsmtSF']], axis=1)
    data1.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));

    fig = plt.figure()
    sns.distplot(trainData['TotalBsmtSF'], fit = stats.norm);
    fig = plt.figure()
    res = stats.probplot(trainData['TotalBsmtSF'], plot=plt)

    # SalePrice and OverallQual
    data2 = pd.concat([trainData['SalePrice'], trainData['OverallQual']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data2)
    fig.axis(ymin=0, ymax=800000);
    
    # MoSold and SalePrice
    print(trainData.groupby('MoSold')['SalePrice'].count())
    sns.countplot(x='MoSold',data=trainData)

def missing_values(alldata, trainData, testData):

    alldata_na = pd.DataFrame(alldata.isnull().sum(), columns={'missingNum'})
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(alldata)*100
    alldata_na['existNum'] = len(alldata) - alldata_na['missingNum']
    
    alldata_na['train_notna'] = len(trainData) - trainData.isnull().sum()
    alldata_na['test_notna'] = alldata_na['existNum'] - alldata_na['train_notna'] 
    alldata_na['dtype'] = alldata.dtypes
    
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na


def dataQualityAnalys(DataSet, trainData, testData):
    
    alldata_na = missing_values(DataSet, trainData, testData)
    print(alldata_na)
    
    # poolQC
    print(DataSet['PoolQC'].value_counts())
    poolqc = DataSet.groupby('PoolQC')['PoolArea'].mean()
    
    # has PoolArea value and missing poolQC value
    poolqcna = DataSet[(DataSet['PoolQC'].isnull())& (DataSet['PoolArea']!=0)][['PoolQC','PoolArea']]
    print('Has PoolArea value and missing PoolQc value',poolqcna)
    
    # has PoolQC value and missing PoolArea value
    poolareana = DataSet[(DataSet['PoolQC'].notnull()) & (DataSet['PoolArea']==0)][['PoolQC','PoolArea']]
    print('Has PoolQC value and missing PoolArea value',poolareana)

    # Garage
    a = pd.Series(DataSet.columns)
    GarageList = a[a.str.contains('Garage')].values
    print(GarageList)
    
    # GarageYrBlt
    print(alldata_na.ix[GarageList,:])    
    # -step2:Check GarageArea、GarageCars is 0，other type fill “none”，number column fill “0”
    print(len(DataSet[(DataSet['GarageArea']==0) & (DataSet['GarageCars']==0)]))
    print(len(DataSet[(DataSet['GarageArea']!=0) & (DataSet['GarageCars'].isnull==True)])) 
    
    # Bsmt
    a = pd.Series(DataSet.columns)
    BsmtList = a[a.str.contains('Bsmt')].values
    print(BsmtList)
    
    allBsmtNa = alldata_na.ix[BsmtList,:]
    print(allBsmtNa)
    
    condition = (DataSet['BsmtExposure'].isnull()) & (DataSet['BsmtCond'].notnull())
    DataSet[condition][BsmtList]
    
    condition1 = (DataSet['BsmtCond'].isnull()) & (DataSet['BsmtExposure'].notnull())
    DataSet[condition1][BsmtList]
    
    condition2 = (DataSet['BsmtQual'].isnull()) & (DataSet['BsmtExposure'].notnull())
    DataSet[condition2][BsmtList]
    
    print(DataSet['BsmtFinSF1'].value_counts().head(5))
    print(DataSet['BsmtFinSF2'].value_counts().head(5))
    print(DataSet['BsmtFullBath'].value_counts().head(5))
    print(DataSet['BsmtHalfBath'].value_counts().head(5))   
    print(DataSet['BsmtFinType1'].value_counts().head(5))     
    print(DataSet['BsmtFinType2'].value_counts().head(5))
    
    # MasVnrType/MasVnrArea	
    print(DataSet[['MasVnrType', 'MasVnrArea']].isnull().sum())
    print(len(DataSet[(DataSet['MasVnrType'].isnull())& (DataSet['MasVnrArea'].isnull())])) # 23   
    print(len(DataSet[(DataSet['MasVnrType'].isnull())& (DataSet['MasVnrArea'].notnull())]))   
    print(len(DataSet[(DataSet['MasVnrType'].notnull())& (DataSet['MasVnrArea'].isnull())]))    
    print(DataSet['MasVnrType'].value_counts())    
    MasVnrM = DataSet.groupby('MasVnrType')['MasVnrArea'].median()
    print(MasVnrM)
    mtypena = DataSet[(DataSet['MasVnrType'].isnull())& (DataSet['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
    print(mtypena)
    
    DataSet[(DataSet['MasVnrType']=='None')&(DataSet['MasVnrArea']!=0)][['MasVnrType','MasVnrArea']]
    DataSet[DataSet['MasVnrType']=='None'][['MasVnrArea']]['MasVnrArea'].value_counts()
    
    # MSSubClass/MSZoning
    print(DataSet[DataSet['MSSubClass'].isnull() | DataSet['MSZoning'].isnull()][['MSSubClass','MSZoning']])
    pd.crosstab(DataSet.MSSubClass, DataSet.MSZoning)
    
    # LotFrontage
    print(DataSet[["LotFrontage", "Neighborhood"]].isnull().sum())
    print(DataSet["LotFrontage"].value_counts().head(5))     
    DataSet["LotFrontage"] = DataSet.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    DataSet["Neighborhood"].value_counts()
    
    # others feature
    others = ['Functional', 'Utilities', 'SaleType', 
              'Electrical', 'FireplaceQu', 'Alley',
              'Fence', 'MiscFeature', 'KitchenQual',
              'LotFrontage', 'Exterior1st', 'Exterior2nd']
    print(DataSet[others].isnull().sum())
    print(DataSet['Functional'].value_counts().head(5))
    print(DataSet['Utilities'].value_counts().head(5))
    print(DataSet['SaleType'].value_counts().head(5))
    print(DataSet['Electrical'].value_counts().head(5))
    print(DataSet["Fence"].value_counts().head(5))
    print(DataSet["MiscFeature"].value_counts().head(5)) 
    print(DataSet['KitchenQual'].value_counts().head(5)) 
    print(DataSet['Exterior1st'].value_counts().head(5)) 
    print(DataSet['Exterior2nd'].value_counts().head(5)) 
    print(DataSet['FireplaceQu'].value_counts().head(5))
    print(DataSet['Alley'].value_counts().head(5))
    
    # remove duplitcate value
    DataSet[DataSet[DataSet.columns].duplicated()==True]
    

def dataPreDeal(DataSet, trainData, testData):
    
    alldata_na = missing_values(DataSet, trainData, testData)
    
    # pool
    poolqcna = DataSet[(DataSet['PoolQC'].isnull()) & (DataSet['PoolArea'] != 0)][['PoolQC','PoolArea']]
    areamean = DataSet.groupby('PoolQC')['PoolArea'].mean()
    
    for i in poolqcna.index:
        v = DataSet.loc[i, ['PoolArea']].values
        print(type(np.abs(v-areamean)))
        DataSet.loc[i,['PoolQC']] = np.abs(v-areamean).astype('float64').argmin()
    
    DataSet['PoolQC'] = DataSet["PoolQC"].fillna("None")
    DataSet['PoolArea'] = DataSet["PoolArea"].fillna(0)
    
    # Garage
    DataSet[['GarageCond','GarageFinish','GarageQual','GarageType']] = DataSet[['GarageCond','GarageFinish','GarageQual','GarageType']].fillna('None')
    DataSet[['GarageCars','GarageArea']] = DataSet[['GarageCars','GarageArea']].fillna(0)
    DataSet['Electrical'] = DataSet['Electrical'].fillna( DataSet['Electrical'].mode()[0])

    # Bsmt
    a = pd.Series(DataSet.columns)
    BsmtList = a[a.str.contains('Bsmt')].values
    
    condition = (DataSet['BsmtExposure'].isnull()) & (DataSet['BsmtCond'].notnull()) 
    DataSet.ix[(condition),'BsmtExposure'] = DataSet['BsmtExposure'].mode()[0]
    
    condition1 = (DataSet['BsmtCond'].isnull()) & (DataSet['BsmtExposure'].notnull()) 
    DataSet.ix[(condition1),'BsmtCond'] = DataSet.ix[(condition1),'BsmtQual']
    
    condition2 = (DataSet['BsmtQual'].isnull()) & (DataSet['BsmtExposure'].notnull()) 
    DataSet.ix[(condition2),'BsmtQual'] = DataSet.ix[(condition2),'BsmtCond']
    
    # BsmtFinType1/BsmtFinType2
    condition3 = (DataSet['BsmtFinType1'].notnull()) & (DataSet['BsmtFinType2'].isnull())
    DataSet.ix[condition3,'BsmtFinType2'] = 'Unf'
    
    allBsmtNa = alldata_na.ix[BsmtList,:]
    allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype'] == 'object'].index
    allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype'] != 'object'].index
    DataSet[allBsmtNa_obj] =DataSet[allBsmtNa_obj].fillna('None')
    DataSet[allBsmtNa_flo] = DataSet[allBsmtNa_flo].fillna(0) 
    
    # MasVnr
    MasVnrM = DataSet.groupby('MasVnrType')['MasVnrArea'].median()
    mtypena = DataSet[(DataSet['MasVnrType'].isnull()) & (DataSet['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
    
    for i in mtypena.index:
        v = DataSet.loc[i,['MasVnrArea']].values
        DataSet.loc[i,['MasVnrType']] = np.abs(v-MasVnrM).astype('float64').argmin()
    
    DataSet['MasVnrType'] = DataSet["MasVnrType"].fillna("None")
    DataSet['MasVnrArea'] = DataSet["MasVnrArea"].fillna(0)
    
    # MS
    DataSet["MSZoning"] = DataSet.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))
    
    # LotFrontage
    x = DataSet.loc[DataSet["LotFrontage"].notnull(), "LotArea"]
    y = DataSet.loc[DataSet["LotFrontage"].notnull(), "LotFrontage"]
    t = (x <= 25000) & (y <= 150)
    p = np.polyfit(x[t], y[t], 1)
    DataSet.loc[DataSet['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, DataSet.loc[DataSet['LotFrontage'].isnull(), 'LotArea'])
    
    # others
    DataSet['KitchenQual'] = DataSet['KitchenQual'].fillna(DataSet['KitchenQual'].mode()[0]) 
    DataSet['Exterior1st'] = DataSet['Exterior1st'].fillna(DataSet['Exterior1st'].mode()[0])
    DataSet['Exterior2nd'] = DataSet['Exterior2nd'].fillna(DataSet['Exterior2nd'].mode()[0])
    DataSet["Functional"] = DataSet["Functional"].fillna(DataSet['Functional'].mode()[0])
    DataSet["SaleType"] = DataSet["SaleType"].fillna(DataSet['SaleType'].mode()[0])
    DataSet["Utilities"] = DataSet["Utilities"].fillna(DataSet['Utilities'].mode()[0])
    
    DataSet[["Fence", "MiscFeature"]] = DataSet[["Fence", "MiscFeature"]].fillna('None')
    DataSet['FireplaceQu'] = DataSet['FireplaceQu'].fillna('None')
    DataSet['Alley'] = DataSet['Alley'].fillna('None')
    DataSet.isnull().sum()[DataSet.isnull().sum() > 0]
    
    # GarageYrBlt 159
    year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
    # map years
    DataSet.GarageYrBlt = DataSet.GarageYrBlt.map(year_map)
    DataSet['GarageYrBlt']= DataSet['GarageYrBlt'].fillna('None')
    
    return DataSet

def exceptionValueDeal(DataSet, trainData):

    plt.figure(figsize=(8,6))
    plt.scatter(trainData.GrLivArea,trainData.SalePrice)
    plt.show()
    
    # drop exception value
    outliers_id = trainData[(trainData.GrLivArea > 4000) & (trainData.SalePrice < 200000)].index
    print(outliers_id)
    DataSet = DataSet.drop(outliers_id)
    Y = trainData.SalePrice.drop(outliers_id)
        
    plt.figure(figsize=(8,6))
    plt.scatter(trainData.TotalBsmtSF, trainData.SalePrice)
    plt.show()    
    
    train_now = pd.concat([DataSet.iloc[:1458,:], Y], axis=1)
    test_now = DataSet.iloc[1458:,:]
    train_now.to_csv('../data/train_afterclean.csv')
    test_now.to_csv('../data/test_afterclean.csv')

def dataTransform():
    
    # read clean data
    train = pd.read_csv("../data/train_afterclean.csv")
    test = pd.read_csv("../data/test_afterclean.csv")
    alldata = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
    alldata.shape
    
    
if __name__ == '__main__':
    
    DataSet, trainData, testData = loadDataSet()
    #correlationAnalys(trainData)
    #distributionAnalys(trainData)
    #dataQualityAnalys(DataSet, trainData, testData)
    DataSet = dataPreDeal(DataSet, trainData, testData)
    exceptionValueDeal(DataSet, trainData)