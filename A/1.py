# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 2. Load dataset
df = pd.read_csv("heart.csv")

# 3. Data cleaning
df.drop_duplicates(inplace=True)

# 4. Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 5. Data transformation (Normalization)
scaler = MinMaxScaler()
df[df.select_dtypes(include='number').columns] = scaler.fit_transform(
    df.select_dtypes(include='number')
)

# 6. Split dataset (75% training, 25% testing)
train, test = train_test_split(
    df, test_size=0.25, random_state=42
)

# 7. Display results
print("Original Dataset:", df.shape)
print("Training Data:", train.shape)
print("Testing Data:", test.shape)
print(df.head())
