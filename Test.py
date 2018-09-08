import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

ds = pd.DataFrame(
    [[1, "bag", "Lee", 1, 100], [2, "Leg", "Tom", 2, 220], [2, "bag", "Max", 1, 50], [3, "Tog", "Lim", 3, 1000]],
    columns=["Invoice", "Goods", "Consumer", "Quantity", "Amount"])
ds.unstack()

group = ds.groupby(["Invoice", "Goods"])["Quantity"].sum()
# group = ds.groupby(["Invoice", "Goods"])["Quantity", "Amount"].sum()
# group = ds.groupby(["Invoice", "Goods", "Consumer"])["Quantity"].sum()

ugroup = group.unstack().reset_index().fillna(0)
ugroup = ugroup.set_index("Invoice")
ugroup = ugroup.applymap(lambda x: 1 if x > 0 else 0)

fi = apriori(ugroup, 0.1, True)
r = association_rules(fi, metric="lift")
