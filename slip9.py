items = ['item1', 'item2', 'item3', 'item4']

transactions = [['item1', 'item2', 'item3'],
                ['item2', 'item3'],
                ['item1', 'item2', 'item4'],
                ['item1', 'item4'],
                ['item2', 'item3', 'item4'],
                ['item1', 'item3', 'item4'],
                ['item1', 'item2'],
                ['item1', 'item3'],
                ['item3', 'item4'],
                ['item2', 'item4']]

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


te = TransactionEncoder()
te_ary = te.fit_transform(transactions)


df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)


association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)


print("Frequent itemsets:")
print(frequent_itemsets)
print("\nAssociation rules:")
print(association_rules)

