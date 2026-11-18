
### 7. Naive Bayes

**Bayes Theorem**
- Calculates probability using prior knowledge and new evidence.
- Formula: `P(A|B) = [P(B|A) × P(A)] / P(B)`
- Used in probabilistic classification.

**Naive Bayes Classifier**
- Supervised Machine Learning algorithm.
- Based on Bayes Theorem.
- Assumes features are conditionally independent.
- Used for spam detection and text classification.

**Gaussian Naive Bayes**
- Used for continuous numerical data.
- Assumes features follow a Gaussian (Normal) distribution.
- Python: `GaussianNB()`

**Multinomial Naive Bayes**
- Used for discrete data such as word counts.
- Commonly used in text classification.
- Python: `MultinomialNB()`

**CountVectorizer**
- Converts text into numerical word-count vectors.
- Creates a vocabulary of unique words.
- Represents each document using word frequencies.
- Python: `CountVectorizer()`
- Method: `fit_transform()`

**TF-IDF (Term Frequency–Inverse Document Frequency)**
- Measures the importance of words in a document.
- Reduces the weight of commonly occurring words.
- Gives higher importance to distinctive words.
- Python: `TfidfVectorizer()`
- Method: `fit_transform()`
