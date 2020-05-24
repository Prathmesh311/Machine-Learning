# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:12:48 2020

@author: Psbho
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv("Churn_Modelling.csv")
x= dataset.iloc[:, 3:13].values
y= dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1]= labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2 = LabelEncoder()
x[:, 2]= labelencoder_x2.fit_transform(x[:, 2])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
x = x[:, 1:]


#splitting data n training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


#Fitting XGBoost to training set
from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(x_train, y_train)

#predicting result
y_pred=classifier.predict(x_test)

#claculating confusion metrics
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= classifier, X= x_train, y= y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())





