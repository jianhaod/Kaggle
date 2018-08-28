# [**Digit Recognizer**](https://www.kaggle.com/c/digit-recognizer)

## CopyRight & Author

* Name: Jianhaod
* Email: daipku@163.com 
* Github Repo: https://github.com/jianhaod/Kaggle 
* Kaggle Profile: [id:[Krad](https://www.kaggle.com/daipku)] https://www.kaggle.com/daipku

## Project Analyse

* Getting Started Prediction Competition
* MNIST Program: Typical image detect program, using CNN and other algorthims resolve    
* Algorthims: `SVM`, `RandomForest`

```
Steps: 
1. Data prepare
a) Download csv data files and load data
b) Preview data; understanding the meaning of each column, the type objects of each column, the format of data

2. Model Select and Ensmble  
a) Using SVM classifier detect image data
b) Using RandomForest classifier detect image data  
c) 

```

## 1. Data Analysis

### a) data load

```python
import pandas as pd

def loadDataSet():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    trainData = dataTrain.values[:, 1:]
    trainLabel = dataTrain.values[:, 0]
    testData = dataTest.values[:, :]
    return trainData, trainLabel, testData
```

### b) Preview data  
* PCA dimensionality reduction   
* Transform training data and testing data into npy and low-dimension type   

```
from sklearn.decomposition import PCA

def dataPCA(X_train, X_test, COMPONENT_NUM):
    trainData = np.array(X_train)
    testData = np.array(X_test)

    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(trainData)

    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)

    return pcaTrainData, pcaTestData
```


## 2. Model Select and Ensmble  

### a) Using `SVM` classifier  

```python
from sklearn.svm import SVC

def svmClassifier(trainData, trainLabel):
    svmClf = SVC(C=4, kernel='rbf')
    svmClf.fit(trainData, trainLabel)
    
	return svmClf
```

### b) Using `` classifier

```python
from sklearn.ensemble import RandomForestClassifier

def randomforestClassifier(trainData, trainLabel):
    rfClf = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_split=2,
                                    min_samples_leaf=1,random_state=34)
    rfClf.fit(trainData, trainLabel)

    return rfClf
```
