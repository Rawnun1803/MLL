import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and preprocess data
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                 names=["label", "message"], encoding="latin-1")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df["message"]
y = df["label"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB()
}

results = {}

for name, model in models.items():

    pipe = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", model)
    ])

    # 3. Train and evaluate
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    # 4. Cross-validation
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print("Cross Validation Accuracy:", scores.mean())

    # 5. Hyperparameter tuning
    grid = GridSearchCV(pipe, {"classifier__alpha": [0.1, 0.5, 1.0]},
                        cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    pred = grid.predict(X_test)
    print("Best Parameters:", grid.best_params_)
    print("Tuned Accuracy:", accuracy_score(y_test, pred))

    results[name] = grid.best_score_

# Compare tuned models
print("\nBest Model:", max(results, key=results.get))