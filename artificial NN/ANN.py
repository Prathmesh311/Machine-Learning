# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:13:29 2020

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

#onehotencoder = OneHotEncoder(categories[1])
#x= onehotencoder.fit_transform(x).toarray()


#splitting data n training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

#importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier = Sequential()

#Adding input layer and first hiddden layer
classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))

#Adding Second hidden layer
classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))

#Adding output layer
classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

#Compiling ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Fitting training set to ANN
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

#Predicting the test set result
y_pred= classifier.predict(x_test)
y_pred= (y_pred> 0.5)

#making confusion metrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)




