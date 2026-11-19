
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Heart.csv")

# Data Cleaning
data = data.drop_duplicates()
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Handle Missing Values
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

# Train-Test Split
train, test = train_test_split(
    data, test_size=0.25, random_state=42
)

# Standardization
cols = train.select_dtypes(include='number').columns

scaler = StandardScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

# Display Results
print("Training Data:", train.shape)
print("Testing Data:", test.shape)
print(train.head())
