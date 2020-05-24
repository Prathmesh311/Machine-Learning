# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:55:57 2020

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

#Training decision tree regression model
from sklearn.tree import DecisionTreeRegressor
dc=DecisionTreeRegressor(random_state=0)
dc.fit(x,y)

#predicting new result
print(dc.predict([[6.5]]))

#visualizing decision tree regression result
x_grid=np.arange(min(x), max(x), 0.1 )
x_grid= x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, dc.predict(x_grid), color="blue")
plt.title("Truth or bluff (polinomial regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()
