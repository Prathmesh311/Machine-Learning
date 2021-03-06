# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:16 2020

@author: Psbho
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#Training KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(x_train, y_train)

#predicting result
y_pred=classifier.predict(x_test)

#claculating confusion metrics
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#ploting the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= x_train, y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:, 0].min() - 1, stop= x_set[:, 0].max() + 1),
                   np.arange(start= x_set[:, 1].min() - 1, stop= x_set[:, 1].max() + 1),)
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75, cmap=ListedColormap(("red", "green")) )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(("red", "green"))(i), label=j)
plt.title("K-NN (training set)")
plt.xlabel("age")
plt.ylabel("Salary")
plt.legend()
plt.show()

#ploting the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= x_test, y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:, 0].min() - 1, stop= x_set[:, 0].max() + 1),
                   np.arange(start= x_set[:, 1].min() - 1, stop= x_set[:, 1].max() + 1))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75, cmap=ListedColormap(("red", "green")) )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(("red", "green"))(i), label=j)
plt.title("K-NN (test set)")
plt.xlabel("age")
plt.ylabel("Salary")
plt.legend()
plt.show()
