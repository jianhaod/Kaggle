# [**Titanic**](https://www.kaggle.com/c/titanic)

## CopyRight & Author

* Name: Jianhaod
* Email: daipku@163.com 
* Github Repo: https://github.com/jianhaod/Kaggle 
* Kaggle Profile: [id:[krad](https://www.kaggle.com/daipku)] https://www.kaggle.com/daipku

## Project Analyse

* Getting Started Prediction Competition
* Classify Program: predict customer `Survived` or `Not Survived`
* Algorthims: `RandomForest` `Adaboost` `ExtraTrees` `GBDT` `DecisionTree` `KNN` `SVM`

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

3. Model Select
a) According to the objective function, determined project to supervised learning or non-supervised learning,
   Classified program or Regression program etc.
b) Compare each basic model, and then take better models

4. Model Ensmble
a) Use Ensmble method to increase model effect; 
   Include Bagging, Boost, Stacking and Blending etc.

5. Optimize and Hyper Parameter Select
a) Fold training data for cross validation
b) Add or update features to increase model
c) Change model hyper parameters to increase model limitation

```

## 1. Data Analysis

### a) data load

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loadData():   
    train_data = pd.read_csv('../input/train.csv')
    test_data = pd.read_csv('../input/test.csv')
    
    sns.set_style('whitegrid')    
    train_data.head(5)

    train_data.info()
    test_data.info()
    train_data.describe()    
    
    return train_data, test_data
```

![](/Titanic/images/Titanic_top_5.jpg)

### b) data preview

* Data Dictionary  

| Variable | Definition | Key |
| - | - | - |
| survival | Survival           | 0 = No, 1 = Yes |
| pclass   | Ticket class       | 1 = 1st, 2 = 2nd, 3 = 3rd |  
| sex      | Sex                | |
| Age      | Age in years       | |    
| sibsp    | # of siblings / spouses aboard the Titanic   | | 
| parch    | # of parents / children aboard the Titanic   | |
| ticket   | Ticket number           | |   
| fare     | Passenger fare          | |  
| cabin    | Cabin number            | |    
| embarked | Port of Embarkation     | C=Cherbourg, Q=Queenstown, S=Southampton |  

* Training data overview  

![](/Titanic/images/Titanic_traindata_info.JPG)

* Testing data overview
  
![](/Titanic/images/Titanic_testdata_info.JPG)

* Training data describe  

![](/Titanic/images/Titanic_traindata_desc.JPG)

### c) feature preliminary analysis

* Nan feature value filled
use vote mode fill `Embarked` null feature
use `Uknow` fill `Cabin` feature
use `RandomForest` to predict age with feature 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare'

```python
def nanValueDeal(DataSet):   
    DataSet.Embarked[DataSet.Embarked.isnull()] = DataSet.Embarked.dropna().mode().values
    DataSet['Cabin'] = DataSet.Cabin.fillna('Uknow')
    
    AgeDataSet = DataSet[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
    AgeDataSetNotNull = AgeDataSet.loc[(DataSet['Age'].notnull())]
    AgeDataSetIsNull = AgeDataSet.loc[(DataSet['Age'].isnull())]
    X = AgeDataSetNotNull.values[:, 0:-1]
    Y = AgeDataSetNotNull.values[:, -1]    
    
    rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    rfr.fit(X,Y)
    
    PredictAges = rfr.predict(AgeDataSetIsNull.values[:, 0:-1])
    DataSet.loc[DataSet['Age'].isnull(), ['Age']] = PredictAges   
```

* Feature and result correlation analysis
Find out data result correlation with `Sex`, `Pclass`

```python
def dataAnayls(DataSet):  
    DataSet[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
    DataSet[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
```

