import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


TID = {1:["bread","milk"],
 2:["bread","diaper","beer","eggs"],
 3:["milk","diaper","beer","coke"],
 4:["bread","milk","diaper","beer"], 
 5:["bread","milk","diaper","coke"]}
transactions = []
for key, value in TID.items():
    transactions.append(value)


te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


min_sup_values = [0.2, 0.4, 0.6]
for min_sup in min_sup_values:
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    print("Min_sup:", min_sup)
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    print("Association Rules:")
    print(rules)

