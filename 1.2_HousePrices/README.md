# [**House Prices**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## CopyRight & Author

* Name: Jianhaod
* Email: daipku@163.com 
* Github Repo: https://github.com/jianhaod/Kaggle 
* Kaggle Profile: [id:[krad](https://www.kaggle.com/daipku)] https://www.kaggle.com/daipku

## Project Analyse

* Getting Started Prediction Competition
* Regression Program: use Multi-dimension features to predict scale target `prices`
* Algorthims: `LASSO`, `ElasticNet`, `XGBOOST`

```
Steps: 

1. Data Analysis
a) Download csv data files and load data
b) Preview data; understanding the meaning of each column, 
   the type objects of each column, the format of data
c) feature preliminary analysis; using plt tools and statistics methods; 
   find out the correlation between the data and columns for Feature Engineering reference

2. Feature Engineering
a) Feature transform; fill nan data, doc value deal, Timestamp processing, category feature encode, 
   binning/split bin feature, dummy feature, feature scaling, feature normalization
b) Feature select; Base on business scenario and common sense, using sorce rank and statistics methods to select
   most important features   

3. Model Select and Ensmble
a) According to the objective function, determined project to supervised learning or non-supervised learning,
   Classified program or Regression program etc.
b) Compare each basic model, and then take better models
c) Use Ensmble method to increase model effect; 
   Include Bagging, Boost, Stacking and Blending etc.

4. Optimize and Hyper Parameter Select
a) Fold training data for cross validation
b) Add or update features to increase model
c) Change model hyper parameters to increase model limitation

```

## 1. Data Analysis

### a) data load

```python
import pandas as pd

def loadDataSet():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    DataSet = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], 
                        test.loc[:, 'MSSubClass':'SaleCondition']), ignore_index = True)
    DataSet.info()
    DataSet.describe()    
    return DataSet, train, test
```
### b) data preview

![](/1.2_HousePrices/images/HousePrice_data_describe.JPG)

### c) feature preliminary analysis

* Find out all the correlation ship with each features  
Show the correlation ship with each features  
Show the top 10 most import features which impact `SalePrice`  
Show the pairpoint picture with `SalePrice`, `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`

```python
def correlationAnalys(trainData):   
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
```

* Correlation heat map  

![](/1.2_HousePrices/images/HousePrice_correlation_feature.JPG)

* Top 10 features most correlation with SalePrice  

![](/1.2_HousePrices/images/HousePrice_top10_heatmap.JPG)

* Features Pairpoint picture    

![](/1.2_HousePrices/images/HousePrice_pairpoint_pic.png)

* Analysis train data `SalePrice`, `GrliveArea`, `TotalBsmtSF` distribution curve and scatter diagram  

```python
def distributionAnalys(trainData):
    sns.distplot(trainData['SalePrice'], fit = stats.norm)   
    print("Skewness: %f" %trainData['SalePrice'].skew())
    print("Kurtosis: %f" %trainData['SalePrice'].kurt())
    
    data1 = pd.concat([trainData['SalePrice'], trainData['GrLivArea']], axis=1)
    data1.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
       
    fig = plt.figure()
    sns.distplot(trainData['GrLivArea'], fit = stats.norm)

    data1 = pd.concat([trainData['SalePrice'], trainData['TotalBsmtSF']], axis=1)
    data1.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));

    fig = plt.figure()
    sns.distplot(trainData['TotalBsmtSF'], fit = stats.norm);

    data2 = pd.concat([trainData['SalePrice'], trainData['OverallQual']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data2)
    fig.axis(ymin=0, ymax=800000);
   
    fig = plt.figure()
    sns.countplot(x='MoSold',data=trainData)
```

* `SalePrice` distribution curve    

![](/1.2_HousePrices/images/HousePrice_trainData_price_norm.JPG)


## 2. Feature Engineering

### a) Feature transform

* Data clean `Nan` value filled  
`pool`, `Garage*`, `Bsmt*`, `MasVnr*`, `MS*`, `LotFrontage`, and etc feature Nan value filled  
`GarageYrBlt` year feature do value map   

```python
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

    # Bsmt*
    a = pd.Series(DataSet.columns)
    BsmtList = a[a.str.contains('Bsmt')].values
    
    condition = (DataSet['BsmtExposure'].isnull()) & (DataSet['BsmtCond'].notnull()) 
    DataSet.ix[(condition),'BsmtExposure'] = DataSet['BsmtExposure'].mode()[0]
    
    condition1 = (DataSet['BsmtCond'].isnull()) & (DataSet['BsmtExposure'].notnull()) 
    DataSet.ix[(condition1),'BsmtCond'] = DataSet.ix[(condition1),'BsmtQual']
    
    condition2 = (DataSet['BsmtQual'].isnull()) & (DataSet['BsmtExposure'].notnull()) 
    DataSet.ix[(condition2),'BsmtQual'] = DataSet.ix[(condition2),'BsmtCond']
    
    condition3 = (DataSet['BsmtFinType1'].notnull()) & (DataSet['BsmtFinType2'].isnull())
    DataSet.ix[condition3,'BsmtFinType2'] = 'Unf'
    
    allBsmtNa = alldata_na.ix[BsmtList,:]
    allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype'] == 'object'].index
    allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype'] != 'object'].index
    DataSet[allBsmtNa_obj] =DataSet[allBsmtNa_obj].fillna('None')
    DataSet[allBsmtNa_flo] = DataSet[allBsmtNa_flo].fillna(0) 
    
    # MasVnr**
    MasVnrM = DataSet.groupby('MasVnrType')['MasVnrArea'].median()
    mtypena = DataSet[(DataSet['MasVnrType'].isnull()) & (DataSet['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
    
    for i in mtypena.index:
        v = DataSet.loc[i,['MasVnrArea']].values
        DataSet.loc[i,['MasVnrType']] = np.abs(v-MasVnrM).astype('float64').argmin()
    
    DataSet['MasVnrType'] = DataSet["MasVnrType"].fillna("None")
    DataSet['MasVnrArea'] = DataSet["MasVnrArea"].fillna(0)
    
    # MS**
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
```

* Exception Value Deal   
`GrLivArea` exception value drop   
Save Data clean value into new csv file  

```python
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
```

![](/1.2_HousePrices/images/HousePrice_GrlivArea_after.JPG)

### b) Feature Select

* Feature data transform   
Feature attribute construction    
Virtual feature create   
Dummy feature, feature split, feature scaling, feature normalization etc   

```python
def FeatureTransform():
    
    # read clean data
    train = pd.read_csv("../output/train_afterclean.csv")
    test = pd.read_csv("../output/test_afterclean.csv")
    alldata = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
    alldata.shape

    #  ordering feature 
    ordinalList = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'PoolQC', 
                   'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual','BsmtCond']
    ordinalmap = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0}
    for c in ordinalList:
        alldata[c] = alldata[c].map(ordinalmap) 

    # map feature to label value
    alldata['BsmtExposure'] = alldata['BsmtExposure'].map({'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})    
    alldata['BsmtFinType1'] = alldata['BsmtFinType1'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
    alldata['BsmtFinType2'] = alldata['BsmtFinType2'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
    alldata['Functional'] = alldata['Functional'].map({'Maj2':1, 'Sev':2, 'Min2':3, 'Min1':4, 'Maj1':5, 'Mod':6, 'Typ':7})
    alldata['GarageFinish'] = alldata['GarageFinish'].map({'None':0, 'Unf':1, 'RFn':2, 'Fin':3})
    alldata['Fence'] = alldata['Fence'].map({'MnWw':0, 'GdWo':1, 'MnPrv':2, 'GdPrv':3, 'None':4})

    # create bin value
    MasVnrType_Any = alldata.MasVnrType.replace({'BrkCmn': 1,'BrkFace': 1,'CBlock': 1,'Stone': 1,'None': 0})
    MasVnrType_Any.name = 'MasVnrType_Any' 
    SaleCondition_PriceDown = alldata.SaleCondition.replace({'Abnorml': 1,'Alloca': 1,'AdjLand': 1,'Family': 1,'Normal': 0,'Partial': 0})
    SaleCondition_PriceDown.name = 'SaleCondition_PriceDown' 
    alldata = alldata.replace({'CentralAir': {'Y': 1,'N': 0}})
    alldata = alldata.replace({'PavedDrive': {'Y': 1,'P': 0,'N': 0}})
    newer_dwelling = alldata['MSSubClass'].map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})
    newer_dwelling.name= 'newer_dwelling' 
    alldata['MSSubClass'] = alldata['MSSubClass'].apply(str)
    Neighborhood_Good = pd.DataFrame(np.zeros((alldata.shape[0],1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[alldata.Neighborhood=='NridgHt'] = 1
    Neighborhood_Good[alldata.Neighborhood=='Crawfor'] = 1
    Neighborhood_Good[alldata.Neighborhood=='StoneBr'] = 1
    Neighborhood_Good[alldata.Neighborhood=='Somerst'] = 1
    Neighborhood_Good[alldata.Neighborhood=='NoRidge'] = 1
    Neighborhood_Good.name='Neighborhood_Good'
    season = (alldata['MoSold'].isin([5,6,7]))*1 
    season.name='season'
    alldata['MoSold'] = alldata['MoSold'].apply(str)

    # use Qual and Cond create new feature
    # deal OverallQual: split two feature from 5
    overall_poor_qu = alldata.OverallQual.copy()
    overall_poor_qu = 5 - overall_poor_qu
    overall_poor_qu[overall_poor_qu < 0] = 0
    overall_poor_qu.name = 'overall_poor_qu'
    overall_good_qu = alldata.OverallQual.copy()
    overall_good_qu = overall_good_qu - 5
    overall_good_qu[overall_good_qu < 0] = 0
    overall_good_qu.name = 'overall_good_qu'
    
    # deal OverallCond split two feature from 5
    overall_poor_cond = alldata.OverallCond.copy()
    overall_poor_cond = 5 - overall_poor_cond
    overall_poor_cond[overall_poor_cond < 0] = 0
    overall_poor_cond.name = 'overall_poor_cond'
    overall_good_cond = alldata.OverallCond.copy()
    overall_good_cond = overall_good_cond - 5
    overall_good_cond[overall_good_cond<0] = 0
    overall_good_cond.name = 'overall_good_cond'
    
    # deal ExterQual split two feature from 3
    exter_poor_qu = alldata.ExterQual.copy()
    exter_poor_qu[exter_poor_qu < 3] = 1
    exter_poor_qu[exter_poor_qu >= 3] = 0
    exter_poor_qu.name = 'exter_poor_qu'
    exter_good_qu = alldata.ExterQual.copy()
    exter_good_qu[exter_good_qu <= 3] = 0
    exter_good_qu[exter_good_qu > 3] = 1
    exter_good_qu.name = 'exter_good_qu'
    
    # deal ExterCond split two feature from 3
    exter_poor_cond = alldata.ExterCond.copy()
    exter_poor_cond[exter_poor_cond < 3] = 1
    exter_poor_cond[exter_poor_cond >= 3] = 0
    exter_poor_cond.name = 'exter_poor_cond'
    exter_good_cond = alldata.ExterCond.copy()
    exter_good_cond[exter_good_cond <= 3] = 0
    exter_good_cond[exter_good_cond > 3] = 1
    exter_good_cond.name = 'exter_good_cond'
    
    # deal BsmtCond split two feature from 3
    bsmt_poor_cond = alldata.BsmtCond.copy()
    bsmt_poor_cond[bsmt_poor_cond < 3] = 1
    bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
    bsmt_poor_cond.name = 'bsmt_poor_cond'
    bsmt_good_cond = alldata.BsmtCond.copy()
    bsmt_good_cond[bsmt_good_cond <= 3] = 0
    bsmt_good_cond[bsmt_good_cond > 3] = 1
    bsmt_good_cond.name = 'bsmt_good_cond'
    
    # deal GarageQual split two feature from 3
    garage_poor_qu = alldata.GarageQual.copy()
    garage_poor_qu[garage_poor_qu < 3] = 1
    garage_poor_qu[garage_poor_qu >= 3] = 0
    garage_poor_qu.name = 'garage_poor_qu'
    garage_good_qu = alldata.GarageQual.copy()
    garage_good_qu[garage_good_qu <= 3] = 0
    garage_good_qu[garage_good_qu > 3] = 1
    garage_good_qu.name = 'garage_good_qu'
    
    # deal GarageCond split two feature from value 3
    garage_poor_cond = alldata.GarageCond.copy()
    garage_poor_cond[garage_poor_cond < 3] = 1
    garage_poor_cond[garage_poor_cond >= 3] = 0
    garage_poor_cond.name = 'garage_poor_cond'
    garage_good_cond = alldata.GarageCond.copy()
    garage_good_cond[garage_good_cond <= 3] = 0
    garage_good_cond[garage_good_cond > 3] = 1
    garage_good_cond.name = 'garage_good_cond'
    
    # deal kitchenQua split two feature from value 3
    kitchen_poor_qu = alldata.KitchenQual.copy()
    kitchen_poor_qu[kitchen_poor_qu < 3] = 1
    kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
    kitchen_poor_qu.name = 'kitchen_poor_qu'
    kitchen_good_qu = alldata.KitchenQual.copy()
    kitchen_good_qu[kitchen_good_qu <= 3] = 0
    kitchen_good_qu[kitchen_good_qu > 3] = 1
    kitchen_good_qu.name = 'kitchen_good_qu'    

    qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                         exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu, 
                         garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

    # timestamp feature deal
    Xremoded = (alldata['YearBuilt'] != alldata['YearRemodAdd'])*1 
    Xrecentremoded = (alldata['YearRemodAdd'] >= alldata['YrSold'])*1 
    XnewHouse = (alldata['YearBuilt'] >= alldata['YrSold'])*1
    XHouseAge = 2010 - alldata['YearBuilt']
    XTimeSinceSold = 2010 - alldata['YrSold']
    XYearSinceRemodel = alldata['YrSold'] - alldata['YearRemodAdd']
    
    Xremoded.name='Xremoded'
    Xrecentremoded.name='Xrecentremoded'
    XnewHouse.name='XnewHouse'
    XTimeSinceSold.name='XTimeSinceSold'
    XYearSinceRemodel.name='XYearSinceRemodel'
    XHouseAge.name='XHouseAge'
    
    year_list = pd.concat((Xremoded,Xrecentremoded,XnewHouse,XHouseAge,XTimeSinceSold,XYearSinceRemodel),axis=1) 
    
    # use SVM create new feature price_category
    svm = SVC(C=100, gamma=0.0001, kernel='rbf')
    
    pc = pd.Series(np.zeros(train.shape[0]))
    pc[:] = 'pc1'
    pc[train.SalePrice >= 150000] = 'pc2'
    pc[train.SalePrice >= 220000] = 'pc3'
    columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
    X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)
    svm.fit(X_t, pc)
    p = train.SalePrice/100000
    
    price_category = pd.DataFrame(np.zeros((alldata.shape[0],1)), columns=['pc'])
    X_t = pd.get_dummies(alldata.loc[:, columns_for_pc], sparse=True)
    pc_pred = svm.predict(X_t)
    
    price_category[pc_pred == 'pc2'] = 1
    price_category[pc_pred == 'pc3'] = 2
    price_category.name = 'price_category'    

    # discretization
    year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
    # map year
    alldata.YearBuilt = alldata.YearBuilt.map(year_map)
    alldata.YearRemodAdd = alldata.YearRemodAdd.map(year_map)
    
    # deal continues value type feature 
    # sample quantile
    numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index
    # choose top 75%
    t = alldata[numeric_feats].quantile(.75) 
    use_75_scater = t[t != 0].index
    alldata[use_75_scater] = alldata[use_75_scater]/alldata[use_75_scater].quantile(.75)

    # fit feature value to standard
    t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    lam = 0.15
    for feat in t:
        alldata[feat] = boxcox1p(alldata[feat], lam)
    
    # binarylize feature 
    X = pd.get_dummies(alldata)
    X = X.fillna(X.mean())
    
    X = X.drop('Condition2_PosN', axis=1)
    X = X.drop('MSZoning_C (all)', axis=1)
    X = X.drop('MSSubClass_160', axis=1)
    X = pd.concat((X, newer_dwelling, season, year_list ,qu_list, MasVnrType_Any, 
                   price_category, SaleCondition_PriceDown, Neighborhood_Good), axis=1)
    
    def poly(X):   
        areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']  
        t = chain(qu_list.axes[1].get_values(), year_list.axes[1].get_values(), 
                  ordinalList, ['MasVnrType_Any'])        
        for a, t in product(areas, t):
            x = X.loc[:, [a, t]].prod(1) 
            x.name = a + '_' + t
            yield x
    
    XP = pd.concat(poly(X), axis=1) 
    X = pd.concat((X, XP), axis=1)
    X_train = X[:train.shape[0]]
    X_test = X[train.shape[0]:]
    Y= train.SalePrice
    train_now = pd.concat([X_train,Y], axis=1)
    test_now = X_test
    
    train_now.to_csv('../output/train_afterchange.csv')
    test_now.to_csv('../output/test_afterchange.csv')
```

## 3.  Model Select and Ensmble

* Model Select using base model `LASSO MODEL`, `XGBOOST`, `ELASTIC NET`    
Ensmble three model result using parameter `0.72`, `0.14`, `0.14`   

```python
def createPredictModel():
    train = pd.read_csv('../output/train_afterchange.csv')
    test = pd.read_csv('../output/test_afterchange.csv')
    alldata = pd.concat((train.iloc[:,1:-1], test.iloc[:,1:]), ignore_index=True)

    X_train = train.iloc[:,1:-1]
    y = train.iloc[:,-1]
    X_test = test.iloc[:,1:]
       
    #use LASSO MODEL
    clf1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,0.0003,0.0002, 5e-4])
    clf1.fit(X_train, y)
    lasso_preds = np.expm1(clf1.predict(X_test))

    #use ELASTIC NET
    clf2 = ElasticNet(alpha=0.0005, l1_ratio=0.9)
    clf2.fit(X_train, y)
    elas_preds = np.expm1(clf2.predict(X_test))

    # use XGBOOST
    clf3=xgb.XGBRegressor(colsample_bytree=0.4,
                          gamma=0.045, learning_rate=0.07,
                          max_depth=20, min_child_weight=1.5,
                          n_estimators=300, reg_alpha=0.65,
                          reg_lambda=0.45, subsample=0.95)   
    clf3.fit(X_train, y.values)
    xgb_preds = np.expm1(clf3.predict(X_test))
    
    # Finally use model ensemble
    final_result = 0.702*lasso_preds + 0.1496*xgb_preds + 0.1502*elas_preds

    solution = pd.DataFrame({"id":test.index+1461, "SalePrice":final_result}, columns=['id', 'SalePrice'])
    solution.to_csv("../output/SubmitResult.csv", index = False) 
```

## 4. Submit Kaggle result

* Public Leaderboard validation Score
 
![](/1.2_HousePrices/images/HousePrice_submit_result.JPG)

* Competitions result rank in all teams

![](/1.2_HousePrices/images/HousePrice_rank_score.JPG)

