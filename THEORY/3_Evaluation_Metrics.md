
### 3. Evaluation Metrics

**Confusion Matrix**
- Table used to evaluate classification models.
- Compares actual values with predicted values.
- Contains TP, TN, FP and FN.

**True Positive (TP)**
- Model predicts Positive.
- Actual value is Positive.
- Correct positive prediction.

**True Negative (TN)**
- Model predicts Negative.
- Actual value is Negative.
- Correct negative prediction.

**False Positive (FP)**
- Model predicts Positive.
- Actual value is Negative.
- Incorrect positive prediction.
- Also called Type I Error.

**False Negative (FN)**
- Model predicts Negative.
- Actual value is Positive.
- Incorrect negative prediction.
- Also called Type II Error.

**Accuracy**
- Measures overall correct predictions.
- Formula: (TP + TN) / (TP + TN + FP + FN)

**Precision**
- Measures how many predicted positives are actually positive.
- Formula: TP / (TP + FP)

**Recall**
- Measures how many actual positives are correctly identified.
- Formula: TP / (TP + FN)

**F1 Score**
- Harmonic mean of Precision and Recall.
- Formula: 2 × (Precision × Recall) / (Precision + Recall)

**MAE (Mean Absolute Error)**
- Average absolute difference between actual and predicted values.
- Formula: Σ|Actual - Predicted| / n
- Lower MAE indicates better performance.

**MSE (Mean Squared Error)**
- Average of squared prediction errors.
- Formula: Σ(Actual - Predicted)² / n
- Penalizes larger errors more heavily.
- Lower MSE indicates better performance.

**R² Score (Coefficient of Determination)**
- Measures how well the model explains variations in target values.
- Indicates goodness of fit.
- Value closer to 1 indicates better performance.
- 1 = Perfect prediction.
- 0 = Model performs like predicting the mean.
