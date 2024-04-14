import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
dataset = pd.read_csv(url, names=names)


X = dataset.drop('class', axis=1)
y = dataset['class']


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


y_pred = model.predict(X_test)


mse = np.mean((y_test - y_pred) ** 2)
print("MSE:", mse)

