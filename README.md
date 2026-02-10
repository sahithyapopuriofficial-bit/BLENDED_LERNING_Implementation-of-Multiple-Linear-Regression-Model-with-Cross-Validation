# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate input features (all columns except price) and the target (price), then split into training and testing sets.
2. Train a Linear Regression model on the training data.
3. Perform 5-fold cross-validation to evaluate model stability and compute average R².
4. Predict on the test set, calculate MSE, MAE, R², and plot actual vs predicted prices to assess performance.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
data.head()
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print('Name: POPURI SAHITHYA')
print('Reg.No: 212225240106')
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2:{cv_scores.mean():.4f}")
y_pred=model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:

<img width="672" height="137" alt="image" src="https://github.com/user-attachments/assets/5734ac87-1d32-464f-bd18-1b8cc5bcd534" />

<img width="313" height="111" alt="image" src="https://github.com/user-attachments/assets/3c9b07ec-24c2-46fa-9a17-8ee596101249" />

<img width="993" height="685" alt="image" src="https://github.com/user-attachments/assets/2f2d70da-bb27-4852-8e5b-d56669ae2e03" />

## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
