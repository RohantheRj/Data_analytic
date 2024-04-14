from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd


TID = {1:["apple","mango","banana"],
       2:["mango","banana","cabbage","carrots"],
       3:["mango","banana","carrots"],
       4:["mango","carrots"]}


te = TransactionEncoder()
te_ary = te.fit([TID[i] for i in TID]).transform([TID[i] for i in TID])
df = pd.DataFrame(te_ary, columns=te.columns_)


min_sup_values = [0.25, 0.5, 0.75]
for min_sup in min_sup_values:
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    print("Frequent itemsets with min_sup =", min_sup)
    print(frequent_itemsets)
    print("\n")

