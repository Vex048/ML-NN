import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


df=pd.read_csv("datasets/Student_Marks.csv")
data=df.to_numpy()

X_set=data[:,0:-1]
Y_set=data[:,-1]

# scaler= StandardScaler()
# X_scaled=scaler.fit_transform(X_set)


X_train,X_set2,y_train,y_set2 = train_test_split(X_set,Y_set,test_size=0.4)
X_validation,X_test,y_validation,y_test = train_test_split(X_set2,y_set2,test_size=0.5)

# Array for best Polynominal model
cv_bestModel=[]
trainingset_mse=[]
for degree in range(1,7):
    poly=PolynomialFeatures(degree,include_bias=False)
    X_train_poly=poly.fit_transform(X_train)

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X_train_poly)


    model=LinearRegression()
    model.fit(X_scaled,y_train)
    y_predict=model.predict(X_scaled)
    trainingset_mse.append(mean_squared_error(y_train, y_predict) / 2)

    X_cv_mapped = poly.transform(X_validation)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)

    y_predict_validation=model.predict(X_cv_mapped_scaled)

    cv_mse = mean_squared_error(y_validation, y_predict_validation) / 2
    cv_bestModel.append(cv_mse)


print(trainingset_mse)
print("-------------------")
print(cv_bestModel)


degree = np.argmin(cv_bestModel) + 1
print(f" Najmniejsza funckja kosztów ma stopień: {degree}")

poly1=PolynomialFeatures(degree,include_bias=False)
X_test_poly=poly1.fit_transform(X_test)

scaler=StandardScaler()
X_scaled_test=scaler.fit_transform(X_test_poly)

model.fit(X_scaled_test,y_test)
y_predict_test=model.predict(X_scaled_test)

test_mse=mean_squared_error(y_test,y_predict_test)

print(f"Training MSE: {trainingset_mse[degree-1]}")
print(f"Cross Validation MSE: {cv_bestModel[degree-1]}")
print(f"Test MSE: {test_mse}")