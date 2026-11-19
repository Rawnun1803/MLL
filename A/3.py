
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *

# 1. Load Dataset
df = pd.read_csv("Admission_Predict.csv")

# 2. Data Preprocessing
df.columns = df.columns.str.strip()
df.drop(columns=["Serial No."], inplace=True)
df.dropna(inplace=True)

df["Admit"] = (df["Chance of Admit"] >= 0.5).astype(int)

# 3. Separate Input and Output
X = df[['GRE Score', 'TOEFL Score', 'University Rating',
        'SOP', 'LOR', 'CGPA', 'Research']]
y = df["Admit"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Prediction and Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Take Student Input
student = pd.DataFrame([[
    int(input("GRE Score: ")),
    int(input("TOEFL Score: ")),
    int(input("University Rating: ")),
    float(input("SOP: ")),
    float(input("LOR: ")),
    float(input("CGPA: ")),
    int(input("Research (0/1): "))
]], columns=X.columns)

result = model.predict(student)
print("Admission:", "Yes" if result[0] == 1 else "No")

# 8. Visualize Decision Tree
plt.figure(figsize=(20, 12))
plot_tree(model, filled=True, feature_names=X.columns,
          class_names=['Not Admitted', 'Admitted'])
plt.show()
