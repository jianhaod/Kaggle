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

![](/1.2_HousePrices/images/HousePrice_data_describe1.JPG) ![](/1.2_HousePrices/images/HousePrice_data_describe2.JPG)

![](/1.2_HousePrices/images/HousePrice_data_describe3.JPG) ![](/1.2_HousePrices/images/HousePrice_data_describe4.JPG) ![](/1.2_HousePrices/images/HousePrice_data_describe5.JPG) 

### b) data preview

* Data Dictionary 

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


