
### 5. Logistic Regression

**Logistic Regression**
- Supervised Machine Learning algorithm.
- Used for classification problems.
- Predicts the probability of a categorical outcome.
- Commonly used for binary classification.

**Sigmoid Function**
- Converts numerical output into a probability between 0 and 1.
- S-shaped curve.
- Output ranges from 0 to 1.
- Formula: `σ(z) = 1 / (1 + e^(-z))`

**Binary Classification**
- Classifies data into two categories.
- Examples: Yes/No, 0/1, Spam/Ham.
- Output consists of two possible classes.

**Probability and Threshold**
- Probability: Likelihood of belonging to a class.
- Threshold: Cut-off value used to assign a class.
- Default threshold: 0.5.
- Probability >= 0.5 → Class 1.
- Probability < 0.5 → Class 0.
- Python: `predict_proba()` and `predict()`
