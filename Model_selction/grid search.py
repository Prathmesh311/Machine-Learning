# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:08:46 2020

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

#Training kernel svm model
from sklearn.svm import SVC
classifier= SVC(kernel="rbf", random_state=0)
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

#Applying Grid Search to find best search and best parameters
from sklearn.model_selection import GridSearchCV
parameters= [{'C': [1, 10, 100, 1000], 'kernel': ["linear"]}, 
             {'C': [1, 10, 100, 1000], 'kernel': ["rbf"], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
            ]

grid_search= GridSearchCV(estimator= classifier, 
                          param_grid= parameters,
                          scoring= "accuracy",
                          cv= 10,
                          n_jobs= -1)
grid_search= grid_search.fit(x_train, y_train)
best_accuracy= grid_search.best_score_
best_parameters= grid_search.best_params_

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
plt.title("kernel-SVM (training set)")
plt.xlabel("age")
plt.ylabel("Salry")
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
plt.title("kernel-SVM (test set)")
plt.xlabel("age")
plt.ylabel("Salry")
plt.legend()
plt.show()


