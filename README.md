# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
Program to implement the linear regression using gradient descent.

Developed by:NIKESH KUMAR C

RegisterNumber:212223040132
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros (X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X = (data.iloc[1:, :-2].values) 
print (X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print('Name:NIKESH KUMAR C'    )
print('Register No.:212223040132'    )
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression (X1_Scaled, Y1_Scaled)
new_data = np.array ([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="1267" height="223" alt="Screenshot 2025-08-31 130114" src="https://github.com/user-attachments/assets/e1b48f79-160d-4b2d-9b09-fb175c212adf" />
<img width="725" height="740" alt="Screenshot 2025-08-31 130151" src="https://github.com/user-attachments/assets/1e683e67-c25e-4059-9c55-c199fae52a79" />
<img width="227" height="376" alt="Screenshot 2025-08-31 130212" src="https://github.com/user-attachments/assets/bfaabb1d-6d81-4591-bea0-26cc64d67f39" />
<img width="556" height="475" alt="Screenshot 2025-08-31 130300" src="https://github.com/user-attachments/assets/edd4aa93-cd3c-41a3-b58c-bd3942f077a9" />
<img width="203" height="467" alt="Screenshot 2025-08-31 130323" src="https://github.com/user-attachments/assets/4806d107-9a62-4be5-a0f4-187d37139e49" />
<img width="404" height="37" alt="Screenshot 2025-08-31 130338" src="https://github.com/user-attachments/assets/38335ee6-f3fe-4ee6-a56f-f8966b370595" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
