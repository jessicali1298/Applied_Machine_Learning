#%%
import pandas as pd
import numpy as np
root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
path1 = root_path + 'dataset1/ionosphere.data'
path2 = root_path + 'dataset2/adult.data'
path3 = root_path + 'dataset3/breast-cancer-wisconsin.data'
path4 = root_path + 'dataset4/machine.data'

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

#%%

def data_report(input_data, dataset_name):
    print('\n----DATA REPORT OF ',dataset_name, '----')
    
    # 1. check data types of all features of input dataset
    data_types = input_data.dtypes
    print('\nfeature dtypes: ', '\n', data_types)
    
    # 2. check missing values
    missing = np.any(pd.isnull(input_data),axis=1)
    missing_idx = np.where(missing == True)[0]
    print('\ninstances with MISSING values: ', '\n', missing_idx)
    
    # 3. check if there are duplicates
    duplicates = np.where(input_data.duplicated() == True)[0]
    print('\nDUPLICATED instances: ', '\n', duplicates)
    
    # 4. check malformed values (mainly in dataset3 - breast cancer)
    malformed = np.where(input_data == '?')[0]
    print('\nMALFORMED instances: ', '\n', malformed)
    
    # 5. number of pos. and neg. classes
    counts = input_data.iloc[:,input_data.shape[1]-1].value_counts()
    print('\n# of binary classifications: ', '\n', counts)

    # 6. Distribution of numerical features (min, max, mean, range)
    numeric_feature_idx = np.where((data_types == int) | (data_types == float))
    
    numeric_df = input_data.loc[numeric_feature_idx]
    numeric_max = numeric_df.max(axis=1)
    numeric_min = numeric_df.min(axis=1)
    numeric_mean = numeric_df.mean(axis=1)
    numeric_range = numeric_max - numeric_min
    
    # numeric_analysis = np.c_(numeric_max, numeric_min, numeric_mean, numeric_range)
    analysis_df = pd.DataFrame({'max':numeric_max, 'min': numeric_min,
                                'mean':numeric_mean, 'range':numeric_range})
    
    
    print('\n', analysis_df)
    
    
#%%
def clean_data(input_data): 
    # 1. remove missing values
    missing = np.any(pd.isnull(input_data),axis=1)
    missing_idx = np.where(missing == True)[0]
    
    # 2. remove duplicates
    duplicates = np.where(input_data.duplicated() == True)[0]
    
    if missing_idx.size != 0 and duplicates != 0: 
        remove_idx = np.unique(np.concatenate(missing_idx, duplicates))
        input_data = input_data.drop(remove_idx)
    
    elif missing_idx.size != 0:
        input_data = input_data.drop(missing_idx)
        
    elif duplicates.size != 0:
        input_data = input_data.drop(duplicates)
    
    # 3. remove malformed values
    malformed = np.where(input_data == '?')[0]
    input_data = input_data.drop(malformed)
    
    output_data = input_data.reset_index(drop=True)
    
    return output_data
    

#%%
# change categorical variable to numerical
def cat_to_num(input_data):
    output_data = input_data.copy()
    
    data_types = input_data.dtypes
    
    # find where non-numeric variables are
    category_var = np.where((data_types != int) & (data_types != float) & (data_types != bool))[0]
    
    # assign unique number code to categorical variables
    for var in category_var:
        codes, uniques = pd.factorize(input_data.iloc[:,var])
        
        # replace original categorical vars with numeric values
        output_data.iloc[:,var] = codes
    
    # print indices of categorical variables/features
    print('\nIndices of Categorical Variables: ', category_var)
    return output_data

#%%
def data_prep(input_data, dataset_name):
    # generate report for basic understanding of dataset
    data_report(input_data, dataset_name)
    cleaned_data = clean_data(input_data)
    final_data = cat_to_num(cleaned_data)
    print(final_data.iloc[0:9,:])
    return final_data

#%%
dataset1_clean = data_prep(dataset1, 'DATASET1--IONOSPHERE')
dataset2_clean = data_prep(dataset2, 'DATASET2--ADULT')
dataset3_clean = data_prep(dataset3, 'DATASET3--BREAST CANCER')
dataset4_clean = data_prep(dataset4, 'DATASET4--MACHINES')


# dataset1 and dataset2 do not need to be modified anymore after data_prep()

# further modification of dataset3:
# - change 7th feature's datatype from str to int64
dataset3_clean.astype({6: 'int64'})

# - drop 1st feature, which is the sample ID name
dataset3_clean = dataset3_clean.drop(dataset3_clean.columns[0], axis=1)


# further modifications of dataset4:
# - define a binary class for the dataset: PRP>50, PRP<=50
class_0_idx = np.where(dataset4_clean[8] <= 50)[0]
class_1_idx = np.where(dataset4_clean[8] > 50)[0]

classes = np.arange(len(dataset4_clean.index))

np.put(classes, class_0_idx, np.zeros(class_0_idx.size))
np.put(classes, class_1_idx, np.ones(class_1_idx.size))

new_df = pd.DataFrame({8: classes})
dataset4_clean.update(new_df)

# - drop 1st, 2nd, and last features, which are vendor name, model name, and estimated relative performance
dataset4_clean = dataset4_clean.drop(columns = {0,1,9})
print(dataset4_clean.iloc[0:20,:])


# change all output datasets to numpy arrays
dataset1_arr = dataset1_clean.to_numpy()
dataset2_arr = dataset1_clean.to_numpy()
dataset3_arr = dataset1_clean.to_numpy()
dataset4_arr = dataset1_clean.to_numpy()

