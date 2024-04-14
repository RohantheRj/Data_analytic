import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


heights = np.random.normal(170, 10, 10)
weights = np.random.normal(70, 5, 10)


dataset = pd.DataFrame({'Height': heights, 'Weight': weights})


X_train, X_test, y_train, y_test = train_test_split(dataset['Height'], dataset['Weight'], test_size=0.2, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train.values.reshape(-1, 1), y_train)


print('Model Coefficients:', lr_model.coef_)


y_pred = lr_model.predict(X_test.values.reshape(-1, 1))
print('Predicted Weights:', y_pred)

