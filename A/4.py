# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

# 3. Data preprocessing
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove unnecessary columns
df.drop(columns=["ID", "ZIP Code"], errors="ignore", inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# 4. Separate features and target
X = df.drop("Personal Loan", axis=1)
y = df["Personal Loan"]

# 5. Train-Test Split (75:25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Apply Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))