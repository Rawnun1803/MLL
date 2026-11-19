
import pandas as pd
from sklearn.metrics import *

# Create Dummy Dataset
actual = [1]*50 + [0]*450
predicted = [1]*45 + [0]*5 + [1]*55 + [0]*395

# Create and Save Excel File
df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
df.to_excel('covid_dataset.xlsx', index=False)

# Read Excel File
data = pd.read_excel('covid_dataset.xlsx')
y_true = data['Actual']
y_pred = data['Predicted']

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Performance Metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
