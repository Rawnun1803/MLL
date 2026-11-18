
### 10. Model Optimization

**Cross-Validation**
- Technique used to evaluate model performance.
- Divides dataset into multiple subsets (folds).
- K-Fold Cross-Validation divides data into K parts.
- Model trains on K-1 folds and tests on the remaining fold.
- Repeats K times and calculates average performance.
- Python: `cross_val_score()`

**Hyperparameter Tuning**
- Process of finding the best model settings.
- Improves model performance.
- Hyperparameters are set before training.
- Examples: `n_estimators`, `max_depth`, `learning_rate`, `K`.

**GridSearchCV**
- Scikit-learn technique for hyperparameter tuning.
- Tests all specified parameter combinations.
- Uses cross-validation to evaluate combinations.
- Selects the best parameter combination.
- Python: `GridSearchCV()`
- Important attributes: `best_params_`, `best_score_`, `best_estimator_`

**Model Comparison**
- Evaluates multiple ML models on the same dataset.
- Compares performance using evaluation metrics.
- Classification: Accuracy, Precision, Recall, F1 Score.
- Regression: MAE, MSE, R² Score.
- Helps select the best-performing model.
