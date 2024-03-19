import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import math

df=pd.read_csv("datasets/Student_Marks.csv")
data=df.to_numpy()
#data=np.delete(data,0,1)
Y_train=np.delete(data,0,1)
Y_train=np.delete(Y_train,0,1)
X_train=np.delete(data,2,1)
# print(X_train)
# print(Y_train)
# print(X_train.shape)
# plt.scatter(X_train,Y_train)
# plt.xlabel("Długość doświadczenia")
# plt.ylabel("Wynagrodzenie")
#plt.show()

def compute_cost(x,y,w,b):
    total_cost=0
    rows=x.shape[0]
    for i in range(rows):
        total_cost=total_cost+ ((np.dot(x[i],w)+b)-y[i])**2
    total_cost=total_cost/(2*rows)
    return total_cost

def compute_gradient(x,y,w,b):
    rows,cols=x.shape
    dj_dw=np.zeros((cols,))
    dj_db=0
    # print("X[i]: ", x[0])
    # print("w: ", w)
    # print("NP.dot : ",np.dot(x[0],w))
    for i in range(rows):
        error=(np.dot(x[i],w)+b)-y[i]
        for j in range(cols):
            dj_dw[j]=dj_dw[j] + error*x[i][j]
        dj_db=dj_db + error
    dj_dw=dj_dw/rows
    dj_db=dj_db/rows
    return dj_dw,dj_db


def gradient_descent(x,y,w_in,b_in,compute_cost,compute_gradient,iterations,alfa):
    
    w = deepcopy(w_in)  
    b = b_in
    Jcost=[]

    for i in range(iterations):
        dw,db=compute_gradient(x,y,w,b)

        w=w-alfa*dw
        b=b-alfa*db
        Jcost.append(compute_cost(x,y,w,b))

        if i% math.ceil(iterations / 10) == 0:
            print(f"Iteration {i}: Cost {Jcost[-1]} ") 
    return w,b,Jcost

m,n=X_train.shape
w_in=np.zeros((n,))
b_in=0
alfa=0.01
iterations=10000
new_w,new_b,Jcost=gradient_descent(X_train,Y_train,w_in,b_in,compute_cost,compute_gradient,iterations,alfa)   
print(f"Oto nowe w: {new_w}, a oto nowe b: {new_b}")
Jcost=np.array(Jcost)
# plt.plot(Jcost,X_train[0])
# plt.show()
    

# X_new=np.delete(data,1,1)
# w1=np.delete(new_w,1,1)
# plt.scatter(X_new,Y_train)
# plt.plot(X_new,(w1*X_new+new_b))
# plt.show()

print(f"Predykcja oceny dla 3 oraz 4.508: {new_w[0]*3 + new_w[1]*4.508+new_b}")
print(f"Predykcja oceny dla 6 oraz 7.711: {new_w[0]*6 + new_w[1]*7.711+new_b}")
print(f"Predykcja oceny dla 3 oraz 6.063: {new_w[0]*3 + new_w[1]*6.063+new_b}")

