# [**House Prices**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## CopyRight & Author

* Name: Jianhaod
* Email: daipku@163.com 
* Github Repo: https://github.com/jianhaod/Kaggle 
* Kaggle Profile: [id:[krad](https://www.kaggle.com/daipku)] https://www.kaggle.com/daipku

## Project Analyse

* Getting Started Prediction Competition
* Regression Program: 
* Algorthims: 

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

