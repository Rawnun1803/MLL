# Confusion Matrix values
TP = 45
FP = 55
FN = 5
TN = 395

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Display results
print("Confusion Matrix:")
print([[TP, FN], [FP, TN]])

print("Accuracy:", accuracy * 100, "%")
print("Precision:", precision * 100, "%")
print("Recall:", recall * 100, "%")
print("F1 Score:", f1 * 100, "%")
