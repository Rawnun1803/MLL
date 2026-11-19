import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("temperatures.csv")

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

df[months] = df[months].fillna(df[months].mean())

X = df[["YEAR"]]
X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

actual, predicted = [], []
models = {}

for month in months:
    y = df[month]
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    actual.extend(y_test)
    predicted.extend(pred)

    models[month] = model
    print(month, "Temperature:", round(model.predict([[df["YEAR"].max() + 1]])[0], 2), "°C")

print("\nMSE:", mean_squared_error(actual, predicted))
print("MAE:", mean_absolute_error(actual, predicted))
print("R2 Score:", r2_score(actual, predicted))

# Visualize January temperature regression
plt.scatter(X, df["JAN"])
plt.plot(X, models["JAN"].predict(X), color="red")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("Linear Regression - January Temperature")
plt.show()