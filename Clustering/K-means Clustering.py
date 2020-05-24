# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:35:57 2020

@author: Psbho
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:, [3,4]].values

#Using Elbow method to find optimul number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans= KMeans(n_clusters = i, init = "k-means++", random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()

#Training the K-means model
kmeans= KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans=kmeans.fit_predict(x)

#Visualising clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c="red", label="cluster 1")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c="blue", label="cluster 2")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c="green", label="cluster 3")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c="cyan", label="cluster 4")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c="yellow", label="cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="black", label="centroids")
plt.title("Clusters of customers")
plt.xlabel("annual income")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()





