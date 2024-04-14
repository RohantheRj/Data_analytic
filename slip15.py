import pandas as pd


data = {'No': [1, 2, 3, 4],
        'Company': ['Tata', 'MG', 'Kia', 'Hyundai'],
        'Model': ['Nexon', 'Astor', 'Seltos', 'Creta'],
        'Year': [2017, 2021, 2019, 2015]}

df = pd.DataFrame(data)

df['Company'] = pd.Categorical(df['Company'])
df['Model'] = pd.Categorical(df['Model'])

df['Company'] = df['Company'].cat.codes
df['Model'] = df['Model'].cat.codes

print(df)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)


association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(association_rules)

