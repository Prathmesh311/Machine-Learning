# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:39:59 2020

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
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:, 1:-1].values
y=dataset.iloc[:, -1].values

#Training linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#Training polinomial linear regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(x)
lin_reg_2= LinearRegression()
lin_reg_2.fit(x_poly,y)

#ploting  linear regression model
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Truth or bluff (Linear regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()

#ploting polynomial regression
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg_2.predict(x_poly), color="blue")
plt.title("Truth or bluff (polynomial regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()

#ploting higher resolution smoother curve
x_grid=np.arange(min(x), max(x), 0.1 )
x_grid= x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Truth or bluff (polinomial regression)")
plt.xlabel("position level")
plt.ylabel("Salery")
plt.show()

#predicting linear model
print(lin_reg.predict([[6.5]]))

#predicting polynimoal model
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


