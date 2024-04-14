import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {'Position': ['CEO', 'charman', 'director', 'Senior Manager', 'Junior Manager', 'Intern'],
        'Level': [1, 2, 3, 4, 5, 6],
        'Salary': [50000, 80000, 110000, 150000, 200000, 250000]}
df = pd.DataFrame(data)

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

