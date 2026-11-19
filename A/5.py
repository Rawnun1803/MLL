import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

df = pd.read_csv("temperatures.csv")

month = input("Enter month: ").upper()
year = int(input("Enter year: "))

X = df[['YEAR']]
y = df[month]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Equation: Temperature =", model.intercept_, "+", model.coef_[0], "* Year")

prediction = model.predict(pd.DataFrame({'YEAR': [year]}))
print("Predicted Temperature:", round(prediction[0], 2), "°C")

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title(month + " Temperature Prediction")
plt.legend()
plt.show()
