import pandas as pd

dataset1 = pd.read_csv(r'dataset1/ionosphere.data', header=None)

print(dataset1.iloc[1])

# basic data validity checks:
# 1. missing values?
# 2. malformed values? (incompatible data type, 
#                      strange out-of-range value, 
#                      positive/negative values only, etc.)
# 3. duplicates?


# data analytics and distributions
# 1. number of pos. and neg. classifications
# 2. distribution of numerical features
# 3. correlation between features
# 4. scatter plots of pair-wise features look-like for some subset of features
 
