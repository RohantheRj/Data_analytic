from sklearn.linear_model import LinearRegression

mileage = [[10], [20], [30], [40], [50], [60], [70], [80]]
price = [24, 19, 17, 13, 10, 7, 5, 2]

reg = LinearRegression().fit(mileage, price)

print('Intercept:', reg.intercept_)
print('Coefficient:', reg.coef_[0])

new_mileage = [[25], [45], [65]]
predicted_price = reg.predict(new_mileage)

print('Predicted prices:', predicted_price)

