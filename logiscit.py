import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf



df=pd.read_csv("framingham.csv")
df.education=df.education.fillna(df.education.mean())
df.cigsPerDay =df.cigsPerDay.fillna(df.cigsPerDay.mean())
df.BPMeds =df.BPMeds.fillna(df.BPMeds.mean())
df.totChol =df.totChol.fillna(df.totChol.mean())
df.BMI =df.BMI.fillna(df.BMI.mean())
df.heartRate =df.heartRate.fillna(df.heartRate.mean())
df.glucose  =df.glucose.fillna(df.heartRate.mean())
scaler=StandardScaler()
cols_to_scale=['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']  

df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])
print("------------")
data=df.to_numpy()
print(data)


X_set=data[:,0:-1]
Y_set=data[:,-1]


 

X_train,X_test,y_train,y_test = train_test_split(X_set,Y_set,test_size=0.2)

model=LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
yp=model.predict(X_test)

# print(f"Test set without regulartazion{y_test[:20]}")
# print(f"Prediction of model without  :{yp[:20]}")



#Neural network 

modelTF=tf.keras.models.Sequential([
    tf.keras.Input(shape=(15,)),
    tf.keras.layers.Dense(units=10,activation='relu',name="layer1"),
    tf.keras.layers.Dense(units=5,activation='relu',name="layer2"),
    tf.keras.layers.Dense(units=1,activation='sigmoid',name="layer3")
])
# W1,b1 = layer1.get_weights()
# W2,b2 = layer2.get_weights()
# W3,b3 = layer3.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

modelTF.compile(optimizer='SGD',
              loss='binary_crossentropy')
modelTF.fit(X_set,Y_set,epochs=100)
modelTF.evaluate(X_test,y_test)

y_pred1=modelTF.predict(X_test)
y_pred2=[]
for i in y_pred1:
    if i>0.5:
        y_pred2.append(1)
    else:
        y_pred2.append(0)
# print(y_pred2[:20])
# print(y_test[:20])

print("Regresja logistyczna") 
print(classification_report(y_test,yp))
print("Siec neuronowa")
print(classification_report(y_test,y_pred2))