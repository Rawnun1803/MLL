import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load and preprocess dataset
df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# 2. Generate transaction list
transactions = df.values.tolist()
transactions = [[item for item in row if pd.notna(item)] for row in transactions]

# 3. Convert transactions into numerical format
te = TransactionEncoder()
data = te.fit_transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# 4. Apply Apriori and generate rules
frequent_items = apriori(df_encoded, min_support=0.01, use_colnames=True)

for confidence in [0.2, 0.4, 0.6]:
    rules = association_rules(frequent_items, metric="confidence",
                              min_threshold=confidence)

    print("\nMinimum Confidence:", confidence)
    print(rules[["antecedents", "consequents", "support",
                 "confidence", "lift"]].head(10))

    # 5. Visualize rules
    if not rules.empty:
        plt.scatter(rules["support"], rules["confidence"])
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Association Rules - Confidence " + str(confidence))
        plt.show()