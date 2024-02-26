# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the standard Libraries.
2.  Set variables for assigning dataset values.
3.  Import linear regression from sklearn.
4.  Assign the points for representing in the graph.
5.  Predict the regression for the marks by using the representation of the graph.
6.  Hence we obtained the linear regression for the given dataset.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KRISHNARAJ D
RegisterNumber: 212222230070
```
```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("/content/score_updated.csv")
df.head(10)
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x = df.iloc[:,0:1]
y = df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
x_train
y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train,lr.predict(X_train),color='red')
lr.coef_
lr.intercept_
```

## Output:
### DATASET
### df.head()
![ML DATASET](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/1dff08a2-b3df-4af6-9184-5f0c09c72d8a)
### df.tail()
![tail](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/ae408e25-07a8-4a4a-9efd-38e40abba5bb)


### GRAPH OF PLOTTED DATA
![ML PLOTTED DATA](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/5cd75057-401a-450d-934a-20806c4c1adc)
### PERFORMING LINEAR REGRESSION
![LINEAR REGR](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/311618c8-2879-4a65-8599-bdc2bea96288)
### TRAINED DATA
![TRAINED DATA](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/95db7b0e-5a03-43aa-952b-52991ce19f69)
### PREDICTING LINE OF REGRESSION
![PREDICTING LINE OF REG](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/95acbc8b-5cfd-4ef9-acbc-dc7ed5fdf2be)
### COEFFICIENT AND INTERCEPT VALUES
![VALUES ML](https://github.com/KRISHNARAJ-D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559695/d33a2ee4-afa2-4fbc-9d51-a1a93759e543)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
