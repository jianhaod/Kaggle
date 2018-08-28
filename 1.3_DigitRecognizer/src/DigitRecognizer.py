#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: svm.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-08-28
"""

import csv
import time
import pandas as pd
import numpy as np
from numpy import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

def loadDataSet():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
    trainData = train.values[:, 1:]
    trainLabel = train.values[:, 0]
    testData = test.values[:, :]
    return trainData, trainLabel, testData

def dataPCA(X_train, X_test):
    trainData = np.array(X_train)
    testData = np.array(X_test)	
    
    pca = PCA(n_components = 0.95, whiten=True)
    pca.fit(trainData)
    
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)
    
    return pcaTrainData, pcaTestData

def svmClassifier(trainData, trainLabel):

    svmClf = SVC(C=4, kernel='rbf')
    svmClf.fit(trainData, trainLabel)
    
    return svmClf

def randomforestClassifier(trainData, trainLabel):
    
    rfClf = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_split=2, 
                                    min_samples_leaf=1,random_state=34)
    rfClf.fit(trainData, trainLabel)
    return rfClf

def knnClassifer(trainData, trainLabel):
    
    knnClf = KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, trainLabel)

    return knnClf


def saveResult(result, csvName):
     with open(csvName, 'wb') as myFile:
         myWriter = csv.writer(myFile)
         myWriter.writerow(["ImageId", "Label"])
         index = 0
         for i in result:
            tmp = []
            index = index+1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)



if __name__ == '__main__':
    trainData, trainLabel, testData = loadDataSet()
    pcatrainData, pcatestData = dataPCA(trainData, testData)

    #svmClf = svmClassifier(pcatrainData, trainLabel)
    #svmtestLabel = svmClf.predict(pcatestData)
    #saveResult(svmtestLabel, r'./SVM.csv')
    
    #rfClf = randomforestClassifier(pcatrainData, trainLabel)
    #rftestLabel = rfClf.predict(pcatestData)
    #saveResult(rftestLabel, r'./RF.csv')

    knnClf = knnClassifer(pcatrainData, trainLabel)
    knntestLabel = knnClf.predict(pcatestData)
    saveResult(knntestLabel, r'./KNN.csv')

