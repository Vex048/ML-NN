import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv("datasets/Student_Marks.csv")
data=df.to_numpy()
#data=np.delete(data,0,1)
Y_train=np.delete(data,0,1)
Y_train=np.delete(Y_train,0,1)
X_train=np.delete(data,2,1)

scaler= StandardScaler()
#print(Y_train.ravel())
X_train=scaler.fit_transform(X_train)
sgdr=SGDRegressor(max_iter=10000)
sgdr.fit(X_train,Y_train.ravel())
print(sgdr)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
y_pred_sgd = sgdr.predict(X_train) 
y_pred = np.dot(X_train, w_norm) + b_norm  


print(b_norm)
print(w_norm)

print(f"prediction using np.dot() and sgdr.predict match: {(y_pred.ravel() == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{Y_train[:4]}")
# plt.scatter(X_train,Y_train)
# plt.plot(X_train,(w_norm*X_train+b_norm))
# plt.show()


print("--------------------------------------------------")
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y_train)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
y_pred_sgd = lin_reg_2.predict(poly_reg.fit_transform(X_train)) 
#y_pred = np.dot(X_poly, w_norm) + b_norm 

print(b_norm)
print(w_norm)



print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{Y_train[:4]}")