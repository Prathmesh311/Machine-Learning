# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:02:47 2020

@author: Psbho
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:, [3,4]].values

#Using Dendrogram to find optimul number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

#Training Hierarchical model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc= hc.fit_predict(x)

#Visualising clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c="red", label="cluster 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c="blue", label="cluster 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c="green", label="cluster 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c="cyan", label="cluster 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c="yellow", label="cluster 5")
plt.title("Clusters of customers")
plt.xlabel("annual income")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()





