import pandas as pd
import numpy as np

path1 = r'dataset1/ionosphere.data'
path2 = r'dataset2/adult.data'
path3 = r'dataset3/breast-cancer-wisconsin.data'
path4 = r'dataset4/machine.data'

dataset1 = pd.read_csv(path1, header=None)
dataset2 = pd.read_csv(path2, header=None)
dataset3 = pd.read_csv(path3, header=None)
dataset4 = pd.read_csv(path4, header=None)

#%%

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

# generate a data summary report
# remove all missing and duplicate instances

# 1. check missing values along instances (rows)
missing = np.any(pd.isnull(dataset1),axis=1)
missing_idx = np.where(missing == True)
print(missing)
print(missing_idx)

# 2. check data types of all features
data_types = dataset1.dtypes
print(type(data_types[0]))

# 3. duplicate instances?
duplicates = np.where(dataset1.duplicated() == True)
#dataset1 = dataset1.drop(duplicates[0]).to_numpy()
print(dataset1.shape)

# 4. number of pos. and neg. classes
counts = dataset1.iloc[:,dataset1.shape[1]-1].value_counts()
print(counts)

# 5. Distribution of numerical features (min, max, mean, range)
numeric_feature_idx = np.where((data_types == int) | (data_types == float))
print(numeric_feature_idx)

numeric_df = dataset1.loc[numeric_feature_idx]
numeric_max = numeric_df.max(axis=1)
numeric_min = numeric_df.min(axis=1)
numeric_mean = numeric_df.mean(axis=1)
numeric_range = numeric_max - numeric_min

print(numeric_max)
print(numeric_min)
print(numeric_mean)
print(numeric_range)


# 6. correlation between features

#%%

def data_report(input_data, dataset_name):
    print('----DATA REPORT OF ', dataset_name, '----')
    # 1. check data types of all features of input dataset
    data_types = dataset1.dtypes
    print('\nfeature dtypes: ', data_types)
    
    # 2. check missing values
    missing = np.any(pd.isnull(input_data),axis=1)
    missing_idx = np.where(missing == True)
    print('\ninstances with MISSING values: ', missing_idx)
    
    # 3. check if there are duplicates
    duplicates = np.where(input_data.duplicated() == True)
    print('\nDUPLICATED instances: ', duplicates)
    
    # 4. number of pos. and neg. classes
    counts = dataset1.iloc[:,dataset1.shape[1]-1].value_counts()
    print('\n# of binary classifications: ', counts)

    # 5. Distribution of numerical features (min, max, mean, range)
    numeric_feature_idx = np.where((data_types == int) | (data_types == float))
    
    numeric_df = input_data.loc[numeric_feature_idx]
    numeric_max = numeric_df.max(axis=1)
    numeric_min = numeric_df.min(axis=1)
    numeric_mean = numeric_df.mean(axis=1)
    numeric_range = numeric_max - numeric_min
    
#    numeric_analysis = np.c_(numeric_max, numeric_min, numeric_mean, numeric_range)
    analysis_df = pd.DataFrame({'max':numeric_max, 'min': numeric_min,
                                'mean':numeric_mean, 'range':numeric_range})
    
    
    print('\n', analysis_df)
    
    
#%%
def clean_data(input_data): 
    # 1. check missing values
    missing = np.any(pd.isnull(input_data),axis=1)
    missing_idx = np.where(missing == True)
    
    # 2. check if there are duplicates
    duplicates = np.where(input_data.duplicated() == True)
    
    if missing_idx[0].size != 0 and duplicates[0] != 0: 
        remove_idx = np.unique(np.concatenate(missing_idx[0], duplicates[0]))
        input_data = input_data.drop(remove_idx)
    
    elif missing_idx[0].size != 0:
        input_data = input_data.drop(missing_idx[0])
        
    elif duplicates[0] != 0:
        input_data = input_data.drop(duplicates[0])
        
    output_data = input_data.to_numpy()
    
    return output_data
    
#%%
data_report(dataset1, 'Dataset1 - ionosphere')
cleaned_data = clean_data(dataset1)


print(cleaned_data.shape)
print(type(cleaned_data))
print(cleaned_data[0])






