import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load and preprocess data
df = pd.read_csv("Mall_Customers.csv")
df.drop(columns=["CustomerID"], inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X = StandardScaler().fit_transform(X)

# 2. Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 3. Apply clustering algorithms
models = {
    "K-Means": KMeans(n_clusters=5, random_state=42, n_init=10),
    "Agglomerative": AgglomerativeClustering(n_clusters=5)
}

for name, model in models.items():
    train_labels = model.fit_predict(X_train)
    test_labels = model.fit_predict(X_test)

    print("\n", name)
    print("Train Silhouette Score:", silhouette_score(X_train, train_labels))
    print("Test Silhouette Score:", silhouette_score(X_test, test_labels))

    # 4. Cross-validation
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train, test in kf.split(X):
        labels = model.fit_predict(X[test])
        scores.append(silhouette_score(X[test], labels))

    print("Cross Validation Score:", sum(scores) / len(scores))

    # 5. Visualize clusters
    labels = model.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
    plt.title(name + " Clustering")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.show()