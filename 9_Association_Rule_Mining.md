
### 9. Association Rule Mining

**Apriori Algorithm**
- Unsupervised Machine Learning algorithm.
- Used for association rule mining.
- Identifies frequently occurring item combinations.
- Generates association rules.
- Uses minimum support and confidence thresholds.
- Python: `apriori()`, `association_rules()`

**Frequent Itemsets**
- Groups of items that frequently occur together.
- Identified using minimum support.
- Example: Bread and Butter.

**Support**
- Measures how frequently an itemset appears in transactions.
- Formula: `Support(A) = Transactions containing A / Total Transactions`

**Confidence**
- Measures the likelihood of purchasing B when A is purchased.
- Formula: `Confidence(A → B) = Support(A ∪ B) / Support(A)`

**Lift**
- Measures the strength of association between two items.
- Formula: `Lift(A → B) = Confidence(A → B) / Support(B)`
- Lift > 1: Positive association.
- Lift = 1: No association.
- Lift < 1: Negative association.

**Market Basket Analysis**
- Data mining technique used to identify purchasing patterns.
- Finds products frequently purchased together.
- Uses Apriori Algorithm and Association Rules.
- Example: Customers purchasing Bread may also purchase Butter.
