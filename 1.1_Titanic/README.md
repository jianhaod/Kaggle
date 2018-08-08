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

![](/1.1_Titanic/images/Titanic_top_5.jpg)

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

![](/1.1_Titanic/images/Titanic_traindata_info.JPG)

* Testing data overview
  
![](/1.1_Titanic/images/Titanic_testdata_info.JPG)

* Training data describe  

![](/1.1_Titanic/images/Titanic_traindata_desc.JPG)

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

![](/1.1_Titanic/images/Titanic_pclass_sex_bar.JPG)

* `Embarked` correlation with survived

![](/1.1_Titanic/images/Titanic_Embarked.JPG)

* `Parch` and `SibSp` correlation with survived

![](/1.1_Titanic/images/Titanic_Parch_SibSp_pipe.JPG)

* `Parch` and `SibSp` distribution with survived

![](/1.1_Titanic/images/Titanic_Parch_SibSp_bar.JPG)

* `Fare` value distribution

![](/1.1_Titanic/images/Titanic_fare_value_distribution.JPG)

* `Fare` mean and std with survived or not 

![](/1.1_Titanic/images/Titanic_fare_mean_std.JPG)

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

![](/1.1_Titanic/images/Titanic_age_value_distribution.JPG)

* `Age` survived and non-survived distribution

![](/1.1_Titanic/images/Titanic_age_survived_distribution.JPG)

* `Age` survived ratio bar

![](/1.1_Titanic/images/Titanic_age_survived_mean_bar.JPG)

* `Age` split by group bar

![](/1.1_Titanic/images/Titanic_age_group_bar.JPG)

## 2. Feature Engineering  

### a) Feature transform  

* Transform object type feature discrete value to numberic   
map `Embarked`, `Sex` value to one-hot code (feature value only has few kinds)   
Extract `Ticket` pre letter to create new feature   
map `Cabin` , `Ticket_Letter` value with factorize (feature value has a lot of kinds not fit use get_dummies method)     

```python
def qualitativeTransfer(DataSet):    
    embarked_onehot_code = pd.get_dummies(DataSet['Embarked'])
    DataSet = DataSet.join(embarked_onehot_code)
    DataSet.drop(['Embarked'], axis = 1, inplace = True)
    
    sex_onehot_code = pd.get_dummies(DataSet['Sex'])
    DataSet = DataSet.join(sex_onehot_code)
    DataSet.drop(['Sex'], axis = 1, inplace = True)
    
    DataSet['CabinLetter'] = DataSet['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    DataSet['CabinLetter'] = pd.factorize(DataSet['CabinLetter'])[0]    
        
    DataSet['Ticket_Letter'] = DataSet['Ticket'].str.split().str[0]
    DataSet['Ticket_Letter'] = DataSet['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
    DataSet['Ticket_Letter'] = pd.factorize(DataSet['Ticket_Letter'])[0]
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
    # factorize and dummies 
    DataSet['Fare_bin_id'] = pd.factorize(DataSet['FareBinning'])[0] 
    fare_bin_dummies_df = pd.get_dummies(DataSet['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
    DataSet = pd.concat([DataSet, fare_bin_dummies_df], axis=1)
    DataSet.drop(['FareBinning'], axis=1, inplace=True)
```

### b) Feature select  

* Select `Title`, Extract Title information and do transform

```python
def titleTransform(DataSet):
    DataSet['Title'] = DataSet['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
        
    title_map_rule = {}
    title_map_rule.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_map_rule.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_map_rule.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_map_rule.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_map_rule.update(dict.fromkeys(['Mr'], 'Mr'))
    title_map_rule.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    DataSet['Title'] = DataSet['Title'].map(title_map_rule)
    
    title_onehot_code = pd.get_dummies(DataSet['Title'])
    DataSet = DataSet.join(title_onehot_code)
    DataSet.drop(['Title'], axis = 1, inplace = True)

    return DataSet
```

* Select `plcass`, combine `Fare` to define pclass category low, middle, high etc   
create a new feature `Plcass_Fare_Category`, and then do dummy

```python
def pclassCategory(DataSet):
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
    
    return DataSet
```

* Select `Parch` and `SibSp` to compute `Family_Size`   
create a new feature `Family_Size_Category` with Family_Size and then do dummy

```python
def familySize(DataSet):
    def family_size_category(family_size):
        if family_size <= 1:
            return 'Single'
        elif family_size <= 4:
            return 'Small_Family'
        else:
            return 'Large_Family'

    DataSet['Family_Size'] = DataSet['Parch'] + DataSet['SibSp'] + 1
    DataSet['Family_Size_Category'] = DataSet['Family_Size'].map(family_size_category)
    
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    DataSet['Family_Size_Category'] = le_family.transform(DataSet['Family_Size_Category'])
    
    family_size_dummies_df = pd.get_dummies(DataSet['Family_Size_Category'],
                                        prefix=DataSet[['Family_Size_Category']].columns[0])
    DataSet = pd.concat([DataSet, family_size_dummies_df], axis=1)
    
    return DataSet
```

* Select most import feature `Age`  
use series orignal and created features `Embarked`, `Sex`, `Title`, `Fare`, `Pclass`    
and `Name_length`, `Family_Size`, `Family_Size_Category`, `Fare_bin_id`  
use `GradientBoostingRegressor` and `RandomForestRegressor` two models predict the result  

```python
def agePredict(ageTrainData, ageTestData):
    age_X_train = ageTrainData.drop(['Age'], axis = 1)
    age_Y_train = ageTrainData['Age']
    age_X_test = ageTestData.drop(['Age'], axis = 1)

    # model 1 gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(age_X_train, age_Y_train)
    ageTestData.loc[:, 'Age_GB'] = gbm_reg_grid.predict(age_X_test)

    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(age_X_train, age_Y_train)
    ageTestData.loc[:, 'Age_RF'] = rf_reg_grid.predict(age_X_test)

    # two models merge
    ageTestData.loc[:, 'Age'] = np.mean([ageTestData['Age_GB'], ageTestData['Age_RF']])
    ageTestData.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return ageTestData
```


## 3. Model Select   

* Prepare input data, dropout no use feature  
regulariz `Age`, `Fare`, `Name_length`  
split whole feature engineering DataSet into training and testing part  

```python    
def preDataSet(DataSet):    
    # regulariz Age Fare Name_length
    scale_age_fare = preprocessing.StandardScaler().fit(DataSet[['Age','Fare', 'Name_length']])
    DataSet[['Age','Fare', 'Name_length']] = scale_age_fare.transform(DataSet[['Age','Fare', 'Name_length']])

    DataSet.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 
                          'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)
    
    train_data = DataSet[:891]
    test_data = DataSet[891:]
    titanic_train_data_X = train_data.drop(['Survived'],axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'],axis=1)
    titanic_train_data_X.shape
    
    return titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X
```

* Split Train data into several parts, in order to do cross validation   


```python
def get_out_fold(clf, x_train, y_train, x_test, ntrain, ntest):
    SEED = 1 # for reproducibility
    NFOLDS = 6 # set folds for out-of-fold prediction
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
```

* Select basic model `RandomForest` `Adaboost` `ExtraTrees` `GBDT` `DecisionTree` `KNN` `SVM`
do model ensmble using `xgboost`

```python
def modelEnsemble(titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X):   
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    
    # Create 7 baseline learn model randomForest/Adaboost/ExtraTrees/GBDT/DecisionTree/KNN/SVM
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, 
                                    min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors = 2)
    svm = SVC(kernel='linear', C=0.025)
    
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
    
    # ensmble
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    gbm = XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('../output/SubmitResult.csv',index=False,sep=',')
```

## 4. Submit Kaggle result

* Public Leaderboard validation Score
 
![](/1.1_Titanic/images/Titanic_submit_result.JPG)

* Competitions result rank in all teams

![](/1.1_Titanic/images/Titanic_rank_score.JPG)


