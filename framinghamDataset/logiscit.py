import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf



df=pd.read_csv("datasets/framingham.csv")
df.education=df.education.fillna(df.education.mean())
df.cigsPerDay =df.cigsPerDay.fillna(df.cigsPerDay.mean())
df.BPMeds =df.BPMeds.fillna(df.BPMeds.mean())
df.totChol =df.totChol.fillna(df.totChol.mean())
df.BMI =df.BMI.fillna(df.BMI.mean())
df.heartRate =df.heartRate.fillna(df.heartRate.mean())
df.glucose  =df.glucose.fillna(df.heartRate.mean())

#df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])
print("------------")
data=df.to_numpy()


# scaler=StandardScaler()
# cols_to_scale=['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose'] 
# df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])

X_set=data[:,0:-1]
Y_set=data[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X_set,Y_set,test_size=0.2)

# print(X_train.shape)
# print(X_test.shape)
# print(X_validaton.shape)


# for degree in range(1,7):
#     poly=PolynomialFeatures(degree)
#     X_poly=poly.fit_transform(X_set)

#     scaler=StandardScaler()
#     cols_to_scale=['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose'] 
#     df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])

#     X_poly[:, 15:]=scaler.fit_transform(X_poly[:, 15:])

#     X_train,X_set2,y_train,y_set2 = train_test_split(X_poly,Y_set,test_size=0.4)
#     X_test,X_validaton,y_test,y_validaton = train_test_split(X_set2,y_set2,test_size=0.5)

#     model=LogisticRegression()
#     model.fit(X_train,y_train)
#     y_pred2=model.predict(X_test)
#     print(f"Degree: {degree}")
#     print(model.score(X_train,y_train))
#     print(model.score(X_validaton,y_validaton))
#     print(model.score(X_test,y_test))
#     print("----------------------------")
#     print("Accuracy score: ",accuracy_score(y_test,y_pred2))


# print(f"Test set without regulartazion{y_test[:20]}")
# print(f"Prediction of model without  :{yp[:20]}")



#Neural network 

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
model=tf.keras.models.Sequential([
    tf.keras.Input(shape=(15,)),
    tf.keras.layers.Dense(units=200,activation='relu',name="layer1"),
    tf.keras.layers.Dense(units=50,activation='relu',name="layer2"),
    tf.keras.layers.Dense(units=75,activation='relu',name="layer3"),
    tf.keras.layers.Dense(units=20,activation='relu',name="layer4"),
    tf.keras.layers.Dense(units=1,activation='sigmoid',name="layer5")
])

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
model.evaluate(X_train,y_train)
#print("Test set: ")
#modelTF.evaluate(X_test,y_test)

y_pred1=model.predict(X_test)
print(y_pred1[:20])
y_pred2=[]
for i in y_pred1:
    if i>0.5:
        y_pred2.append(1)
    else:
        y_pred2.append(0)
print(y_pred2[:20])
print(y_test[:20])


print("Siec neuronowa")
print(classification_report(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
