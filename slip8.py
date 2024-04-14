import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv('market_basket.csv')


df.dropna(inplace=True)


te = TransactionEncoder()
te_ary = te.fit(df.values).transform(df.values)
df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


print("Dataset information:")
print(df.info())


print("\nFrequent itemsets:")
print(frequent_itemsets)


print("\nAssociation rules:")
print(rules)

