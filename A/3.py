
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
df = pd.read_csv("Admission.csv")

# 3. Data preprocessing
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove unnecessary column
df.drop(columns=["Serial No."], errors="ignore", inplace=True)

# Create target if dataset has Chance of Admit instead of Admitted
if "Admitted" not in df.columns:
    df["Admitted"] = (df["Chance of Admit "] >= 0.5).astype(int)
    df.drop(columns=["Chance of Admit "], inplace=True)

# Label Encoding (if categorical columns exist)
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# 4. Separate features and target
X = df.drop("Admitted", axis=1)
y = df["Admitted"]

# 5. Train-Test Split (75:25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 6. Apply Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
