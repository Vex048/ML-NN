import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf

df=pd.read_csv("datasets/Student_Marks.csv")
data=df.to_numpy()

X_set=data[:,0:-1]
Y_set=data[:,-1]

print(X_set)
print(Y_set)

scaler= StandardScaler()
X_scaled=scaler.fit_transform(X_set)


X_train,X_test,y_train,y_test = train_test_split(X_scaled,Y_set,test_size=0.2)

modelTF=tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(units=30,activation='relu',name="layer1"),
    tf.keras.layers.Dense(units=15,activation='relu',name="layer2"),
    tf.keras.layers.Dense(units=5,activation='relu',name="layer3"),
    tf.keras.layers.Dense(units=20,activation='relu',name="layer4"),
    tf.keras.layers.Dense(units=1,activation='linear',name="layer5")
    ])


modelTF.compile(optimizer="Adam",
              loss="mean_absolute_error",
              metrics=['accuracy'])
modelTF.fit(X_train,y_train,epochs=150)
modelTF.evaluate(X_train,y_train)
y_pred1=modelTF.predict(X_test)
print(y_test[:20])
print(y_pred1[:20])