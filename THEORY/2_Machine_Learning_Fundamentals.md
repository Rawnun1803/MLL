
### 2. Machine Learning Fundamentals

**What is Machine Learning?**
- Branch of Artificial Intelligence (AI).
- Enables computers to learn from data.
- Identifies patterns and makes predictions without explicit programming.

**Supervised vs Unsupervised Learning**
- Supervised Learning:
  - Uses labelled data.
  - Input and output are known.
  - Examples: Classification and Regression.
- Unsupervised Learning:
  - Uses unlabelled data.
  - Finds hidden patterns and groups.
  - Examples: K-Means and Hierarchical Clustering.

**Classification vs Regression**
- Classification:
  - Predicts categorical values.
  - Example: Spam or Not Spam.
- Regression:
  - Predicts continuous numerical values.
  - Example: Temperature or House Price.

**Features and Target Variables**
- Features (X): Input variables used for prediction.
- Target (y): Output variable to be predicted.
- Example: Age and Salary → Loan Approval.

**Training and Testing**
- Training Data: Used to train the ML model.
- Testing Data: Used to evaluate model performance.
- Common split: 75% Training and 25% Testing.
- `train_test_split()`

**Overfitting and Underfitting**
- Overfitting:
  - Model learns training data too closely.
  - Performs well on training data but poorly on testing data.
- Underfitting:
  - Model fails to learn important patterns.
  - Performs poorly on both training and testing data.
- Goal: Achieve good generalization.
