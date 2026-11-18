
### 1. Python Fundamentals

**NumPy**
- Numerical Python library.
- `np.array()`, `np.mean()`, `np.max()`, `np.min()`
- Array operations.

**Pandas**
- Library for data manipulation and analysis.
- `pd.read_csv()`
- `head()`, `tail()`, `info()`, `describe()`

**DataFrames and Series**
- DataFrame: 2D tabular data.
- Series: 1D labelled data.

**Reading CSV Files**
```python
df = pd.read_csv("data.csv")
```

**Data Cleaning**
- Remove duplicates: `drop_duplicates()`
- Handle incorrect or inconsistent data.

**Missing Value Handling**
- Check: `isnull().sum()`
- Remove: `dropna()`
- Replace: `fillna()`

**Data Transformation**
- Normalization
- Standardization
- Encoding categorical data.

**Label Encoding**
- Convert categorical values into numerical values.
- `LabelEncoder()`
- `fit_transform()`

**Feature Selection**
- Select relevant input columns.
- Remove unnecessary features.
- X = Features, y = Target.

**Train-Test Split**
- Divide dataset into training and testing sets.
- Training: 75%
- Testing: 25%
- `train_test_split()`
- `random_state`
