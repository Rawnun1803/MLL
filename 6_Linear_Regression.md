
### 6. Linear Regression

**Simple Linear Regression**
- Supervised Machine Learning algorithm.
- Used to predict continuous numerical values.
- Finds the relationship between one independent variable (X) and one dependent variable (Y).
- Equation: `Y = mX + c`

**Slope and Intercept**
- Slope (m): Change in Y for every unit change in X.
- Intercept (c): Value of Y when X = 0.
- Positive slope: Increasing relationship.
- Negative slope: Decreasing relationship.

**Best Fit Line**
- Straight line that best represents the relationship between data points.
- Minimizes the sum of squared errors (residuals).
- Calculated using the Least Squares Method.

**Prediction**
- Estimates output using the trained regression model.
- Uses `model.predict(X_test)`.
- Predicted values are compared with actual values.

**Regression Visualization**
- Scatter Plot: Displays actual data points.
- Regression Line: Shows the predicted relationship.
- Actual vs Predicted: Compares model predictions with actual values.
- Python: `matplotlib.pyplot.scatter()` and `plt.plot()`
