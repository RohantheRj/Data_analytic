import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


dataset = {
    1: ["eggs", "milk", "bread"],
    2: ["eggs", "apple"],
    3: ["milk", "bread"],
    4: ["apple", "milk"],
    5: ["milk", "apple", "bread"]
}


te = TransactionEncoder()
te_ary = te.fit([transaction for transaction in dataset.values()]).transform([transaction for transaction in dataset.values()])
df = pd.DataFrame(te_ary, columns=te.columns_)


min_sup = 0.4
frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)


print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", association_rules)

