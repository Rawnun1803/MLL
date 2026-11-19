
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Load dataset
data = pd.read_csv("Heart.csv")

# 3. Data Cleaning
print("First few rows:")
print(data.head())

print("Duplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# Remove unwanted column
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# 4. Handle Missing Values
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].median())

print("Missing values after handling:")
print(data.isnull().sum())

# 5. Split Dataset
train_data, test_data = train_test_split(
    data, test_size=0.25, random_state=42
)

# 6. Data Transformation (Standardization)
numerical_features = train_data.select_dtypes(
    include=['int64', 'float64']
).columns

scaler = StandardScaler()

train_data = train_data.copy()
test_data = test_data.copy()

train_data[numerical_features] = scaler.fit_transform(
    train_data[numerical_features]
)

test_data[numerical_features] = scaler.transform(
    test_data[numerical_features]
)

# 7. Display Results
print("Training data size:", train_data.shape)
print("Testing data size:", test_data.shape)

print("Transformed training data:")
print(train_data.head())
