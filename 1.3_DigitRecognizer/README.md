# [**Digit Recognizer**](https://www.kaggle.com/c/digit-recognizer)

## CopyRight & Author

* Name: Jianhaod
* Email: daipku@163.com 
* Github Repo: https://github.com/jianhaod/Kaggle 
* Kaggle Profile: [ID:[Krad](https://www.kaggle.com/daipku)] https://www.kaggle.com/daipku

## Project Analyse

* Getting Started Prediction Competition
* MNIST Program: Typical image detect program, using CNN and other algorthims resolve    
* Algorthims: `SVM`, `RandomForest`, `KNN`, `LaNet`

```
Steps: 
1. Data prepare
a) Download csv data files and load data
b) Preview data; understanding the meaning of each column, the type objects of each column, the format of data

2. Model Select and Ensmble  
a) Using SVM classifier detect image data
b) Using RandomForest classifier detect image data  
c) Using KNN classifier detect image data
d) Using different CNN net(LaNet5/Simple CNN/Complex CNN) detect image data   
e) Using Ensmble merge all predict result   

3. Submit Kaggle result

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

```python
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

### b) Using `RandomForest` classifier

```python
from sklearn.ensemble import RandomForestClassifier

def randomforestClassifier(trainData, trainLabel):
    rfClf = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=34)
    rfClf.fit(trainData, trainLabel)

    return rfClf
```

### c) Using `KNN` classifier  

```python
from sklearn.neighbors import KNeighborsClassifier

def knnClassifer(trainData, trainLabel):
    knnClf = KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, trainLabel)

    return knnClf
```

### d) Using different `CNN` net(LaNet5/Simple CNN/Complex CNN) 

* Train `LaNet5` model and predict test data  

```python
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import adam

def LaNetClassifer(trainData, trainLabel, testData):
    model_name = "LaNet5"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
    model.add(Conv2D(kernel_size=(3, 3), filters=6, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(kernel_size=(5, 5), filters=16, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(kernel_size=(5, 5), filters=120, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(Flatten())
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(trainData, trainLabel, epochs=30, batch_size=64)
    y_pred = model.predict_classes(testData)

    return y_pred
```

![](/1.3_DigitRecognizer/images/LaNetTrain.JPG)  

* Train `Simple CNN` model and predict test data  

```python
def SimpleCnnClassifer(trainData, trainLabel, testData):
    model_name = "CNN"
    model = Sequential()
    
    model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
    model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
    model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim=256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=256, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(trainData, trainLabel, epochs=50, batch_size=128)
    y_pred = model.predict_classes(testData)
   
    return y_pred
```

![](/1.3_DigitRecognizer/images/SimpleCNNTrain.JPG)  

* Train `Complex CNN` model and predict test data  

```python
def ComplexCnnClassifer(trainData, trainLabel, testData):
    model_name = "ComplexCNN"
    
    model = Sequential()
    model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=10, activation='softmax'))
    
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(trainData, trainLabel, epochs=80, batch_size=128)
    y_pred = model.predict_classes(testData)
   
    return y_pred
```

![](/1.3_DigitRecognizer/images/ComplexCNNTrain.JPG)

### e) Using Ensmble merge all predict result

```python
def EnsemleResult(y_all_pred):
    model_name = "Ensemble"
    y_ensem_pred = np.zeros((n_samples_test,))

    for i,line in enumerate(y_all_pred.T):
        y_ensem_pred[i] = np.argmax(np.bincount(line))

    y_ensem_pred = y_ensem_pred.astype("int64")
    output_prediction(y_ensem_pred, model_name)
```

## 3. Submit Kaggle result

* Public Leaderboard validation Score

![](/1.2_HousePrices/images/DigitRecognizer_submit_result.JPG)

