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
use `RandomForest` to predict age with feature `Survived`, `Pclass`, `SibSp`, `Parch`, `Fare`

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

```python
def dataAnayls(DataSet):  
    DataSet[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
    DataSet[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()    
    DataSet[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
    
    sns.countplot('Embarked', hue = 'Survived', data = DataSet)
    plt.title('Embarked and Survived')

    has_sibsp = DataSet[DataSet['SibSp'] != 0]
    no_sibsp = DataSet[DataSet['SibSp'] == 0]
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    has_sibsp['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('Has_SibSp')
    
    plt.subplot(122)
    no_sibsp['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
    plt.xlabel('No_SibSp')
    plt.show()
        
    fig, axis = plt.subplots(1,2,figsize=(18,8))
    DataSet[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax = axis[0])
    axis[0].set_title('Parch and Survived')
    DataSet[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax = axis[1])
    axis[1].set_title('SibSp and Survived')
    plt.show()
    
    plt.figure(figsize = (10, 5))
    DataSet['Fare'].hist(bins = 70)
    plt.title('Fare distribution')
    plt.show()
       
    fare_not_survived = DataSet['Fare'][DataSet['Survived'] == 0]
    fare_survived = DataSet['Fare'][DataSet['Survived'] == 1]
    
    average_fare =  pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare  = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
    average_fare.plot(yerr = std_fare, kind='bar', legend = False)
    plt.title('Fare Mean and std with survived')
    plt.show()         
```

* `Plcass` and `Sex` correlation with survived

![](/Titanic/images/Titanic_pclass_sex_bar.JPG)

* `Embarked` correlation with survived

![](/Titanic/images/Titanic_Embarked.JPG)

* `Parch` and `SibSp` correlation with survived

![](/Titanic/images/Titanic_Parch_SibSp_pipe.JPG)

* `Parch` and `SibSp` distribution with survived

![](/Titanic/images/Titanic_Parch_SibSp_bar.JPG)

* `Fare` value distribution

![](/Titanic/images/Titanic_fare_value_distribution.JPG)

* `Fare` mean and std with survived or not 

![](/Titanic/images/Titanic_fare_mean_std.JPG)

* `Age` correlation with survived analysis

```python
def ageAnayls(DataSet):   
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
    
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    DataSet["Age_int"] = DataSet["Age"].astype(int)
    average_age = DataSet[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
    sns.barplot(x='Age_int', y='Survived', data=average_age)
    plt.show()
    
    bins = [0, 12, 18, 65, 100]
    DataSet['Age_group'] = pd.cut(DataSet['Age'], bins)
    group_age = DataSet.groupby('Age_group')['Survived'].mean().plot.bar()
```

* `Age` value distribution and box-plot show 

![](/Titanic/images/Titanic_age_value_distribution.JPG)

* `Age` survived and non-survived distribution

![](/Titanic/images/Titanic_age_survived_distribution.JPG)

* `Age` survived ratio bar

![](/Titanic/images/Titanic_age_survived_mean_bar.JPG)

* `Age` split by group bar

![](/Titanic/images/Titanic_age_group_bar.JPG)

## 2. Feature Engineering  

### a) Feature transform  

* Transform object type feature discrete value to numberic   
map `Embarked` value to one-hot code (feature value only has few kinds)    
map `Cabin` value with factorize (feature value has a lot of kinds not fit use get_dummies method)     

```python
def qualitativeTransfer(DataSet):    
    embarked_onehot_code = pd.get_dummies(DataSet['Embarked'])
    DataSet = DataSet.join(embarked_onehot_code)
    DataSet.drop(['Embarked'], axis = 1, inplace = True)
    
    DataSet['CabinLetter'] = DataSet['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    DataSet['CabinLetter'] = pd.factorize(DataSet['CabinLetter'])[0]
```

* Transform numberic type feature to target range   
reshape `Age` numberic value range in (-1, 1) with scaler
Binning `Fare` value to 5 box, and then do `dummies` or `factorize`    

```python
from sklearn import preprocessing

def quantitativeTransfer(DataSet):
    scaler = preprocessing.StandardScaler()
    DataSet['Age_scaled'] = scaler.fit_transform(DataSet['Age'].values.reshape(-1, 1))
    
    DataSet['FareBinning'] = pd.qcut(DataSet['Fare'], 5)
    # dummies or factorize 
    #DataSet['Fare_bin_id'] = pd.factorize(DataSet['FareBinning'])[0]   
    fare_bin_dummies_df = pd.get_dummies(DataSet['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
    DataSet = pd.concat([DataSet, fare_bin_dummies_df], axis=1)
```

### b) Feature select  

 



