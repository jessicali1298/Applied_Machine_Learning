#%%
import pandas as pd
import numpy as np
import clean_data_class as cd



root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
path1 = root_path + 'dataset1/ionosphere.data'
path2 = root_path + 'dataset2/adult.data'
path3 = root_path + 'dataset3/breast-cancer-wisconsin.data'
path4 = root_path + 'dataset4/machine.data'
path2_test = root_path + 'dataset2/adult.test'

dataset1 = pd.read_csv(path1, header=None)
dataset2 = pd.read_csv(path2, sep = '\,\s+', engine = 'python', header=None)
dataset3 = pd.read_csv(path3, header=None)
dataset4 = pd.read_csv(path4, header=None)
dataset2_test = pd.read_csv(path2_test, sep = '\,\s+', engine = 'python', header=None)


#%%
# ------------------------------DATA CLEANING----------------------------------

# perform data_prep for all datasets
dc = cd.DataCleaner()

dataset1_clean = dc.data_prep(dataset1, 'DATASET1--IONOSPHERE')
dataset2_clean = dc.data_prep(dataset2, 'DATASET2--ADULT')
dataset2_clean_test = dc.data_prep(dataset2_test, 'DATASET2---ADULT_TEST')
dataset3_clean = dc.data_prep(dataset3, 'DATASET3--BREAST CANCER')
dataset4_clean = dc.data_prep(dataset4, 'DATASET4--MACHINES')


# DATASET1 does not need to be modified anymore after data_prep()

# further modification of DATASET3:
# - change 7th feature's datatype from str to int64
dataset3_clean.astype({6: 'int64'})

# - drop 1st feature, which is the sample ID name
dataset3_clean = dataset3_clean.drop(dataset3_clean.columns[0], axis=1)


# further modifications of DATASET4:
# - define a binary class for the dataset: PRP>50, PRP<=50
class_0_idx = np.where(dataset4_clean[8] <= 50)[0]
class_1_idx = np.where(dataset4_clean[8] > 50)[0]

classes = np.arange(len(dataset4_clean.index))

np.put(classes, class_0_idx, np.zeros(class_0_idx.size))
np.put(classes, class_1_idx, np.ones(class_1_idx.size))

# replace old performance data with newly created binary class
new_df = pd.DataFrame({8: classes})
dataset4_clean.update(new_df)

# - drop 1st, 2nd, and last features, which are vendor name, model name, and estimated relative performance
dataset4_clean = dataset4_clean.drop(columns = {0,1,9})
print(dataset4_clean.iloc[0:20,:])


# change all output datasets to numpy arrays
dataset1_arr = dataset1_clean.to_numpy()
dataset2_arr = dataset1_clean.to_numpy()
dataset2_arr_test = dataset2_clean_test.to_numpy()
dataset3_arr = dataset1_clean.to_numpy()
dataset4_arr = dataset1_clean.to_numpy()


#%%



