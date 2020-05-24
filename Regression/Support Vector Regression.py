# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:41:17 2020

@author: Psbho
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:, 1:-1].values
y=dataset.iloc[:, -1].values

y=y.reshape(len(y),1)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y)

#Training SVR model
from sklearn.svm import SVR
regressor= SVR(kernel="rbf")
regressor.fit(x,y)

#predicting new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

#visualizing the SVR result
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color="blue")
plt.title("Truth or bluff (Support vector regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()

#visualizing SVR result in high resolution
x_grid=np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color="blue")
plt.title("Truth or bluff (Support vector regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()










