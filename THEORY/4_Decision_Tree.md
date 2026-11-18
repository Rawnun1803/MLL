
### 4. Decision Tree

**Decision Tree Classifier**
- Supervised Machine Learning algorithm.
- Used for classification problems.
- Represents decisions in a tree-like structure.
- Root Node: Starting point.
- Decision Node: Splits data based on conditions.
- Leaf Node: Final predicted output.

**Entropy**
- Measures impurity or uncertainty in a dataset.
- Higher entropy = More uncertainty.
- Lower entropy = Less uncertainty.
- Formula: `Entropy = -Σ p(i) log₂ p(i)`

**Gini Index**
- Measures impurity in a dataset.
- Lower Gini Index = Better purity.
- Used to select the best data split.
- Formula: `Gini = 1 - Σ p(i)²`

**Information Gain**
- Measures reduction in entropy after splitting data.
- Higher Information Gain = Better split.
- Used to select the best feature for splitting.
- Formula: `IG = Entropy(parent) - Weighted Entropy(children)`

**Label Encoding**
- Converts categorical values into numerical values.
- Assigns a unique integer to each category.
- Used to prepare categorical data for ML algorithms.
- Python: `LabelEncoder()`
- Method: `fit_transform()`
