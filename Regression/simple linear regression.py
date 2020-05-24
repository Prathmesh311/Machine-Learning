# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:45:50 2020

@author: Psbho
"""

#importing liabraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,-1].values

#spliting dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Training simple linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting test set 
y_pred=regressor.predict(x_test)

#ploting training set result
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Experience vs salery (traning set)")
plt.xlabel("experience")
plt.ylabel("Salery")
plt.show()

#ploting test result
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Experience vs salery (test set)")
plt.xlabel("experience")
plt.ylabel("Salery")
plt.show()









