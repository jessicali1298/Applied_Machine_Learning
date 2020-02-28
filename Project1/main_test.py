#%%
import pandas as pd
import numpy as np
import Data_Cleaner as cd
import Log_Regression as lgr
import Naive_Bayes as nb
import Cross_Validation as cv
import time 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB




root_path = '/Users/liuxijun/Downloads/Applied_Machine_Learning/Project1/'
#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
#root_path = '/Users/kirenrao/Documents/GitHub/Applied_Machine_Learning/Project1/'


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
def split_y(input_data):
    y = dc.extract_y(input_data).to_numpy()
    output_data = input_data.drop(input_data.columns[-1], axis=1)
    
    return y, output_data

def test_diff_a(a_arr, input_data):
    
    log_score_ls = []
    sk_log_score_ls = []
    
    avg_acc_ls = []
    avg_precision_ls = []
    avg_recall_ls = []

    time_array = []
    my_time = 0
    
  
    for j in range(a_arr.shape[0]):
        start =  time.time()
        log_acc, sk_log_acc = cvo.log_k_fold(5,input_data,a_arr[j],0.01, 0)
        
        avg_acc = np.mean(log_acc["accuracy"])
        avg_recall = np.mean(log_acc["recall"])
        avg_precision = np.mean(log_acc["precision"])

        avg_acc_ls.append(avg_acc)
        avg_precision_ls.append(avg_precision)
        avg_recall_ls.append(avg_recall)

        log_score_ls.append(log_acc)
        sk_log_score_ls.append(sk_log_acc)
        my_time = time.time() - start
        time_array.append(my_time)
        print(my_time)
    return log_score_ls, sk_log_score_ls, time_array, avg_acc_ls, avg_precision_ls, avg_recall_ls

def test_diff_iter(iter_arr, input_data):
    log_score_ls = []
    sk_log_score_ls = []
    
    for j in range(iter_arr.shape[0]):
        log_acc, sk_log_acc = cvo.log_k_fold_iter(5, input_data, 0.1, iter_arr[j], 0)
        log_score_ls.append(log_acc)
        sk_log_score_ls.append(sk_log_acc)
        
    return log_score_ls, sk_log_score_ls
#%%
# create DataCleaner object
dc = cd.Data_Cleaner()

# --------------------------DATA CLEANING for LOG REG----------------------------

# prep data for logistic regression
dataset1_clean = dc.data_prep(dataset1, 'DATASET1--IONOSPHERE')
dataset2_clean = dc.data_prep(dataset2, 'DATASET2--ADULT')
dataset2_clean_test = dc.data_prep(dataset2_test, 'DATASET2---ADULT_TEST')
dataset3_clean = dc.data_prep(dataset3, 'DATASET3--BREAST CANCER')
dataset4_clean = dc.data_prep(dataset4, 'DATASET4--MACHINES')


#----------DATASET1 does not need to be modified anymore after data_prep()-------

#-----------------------------DATASET2 MODIFICATION------------------------------
# normalize columns 1,3,11 (0,2,10)
col_1 = dc.normalize(dataset2_clean.iloc[:,1])
col_3 = dc.normalize(dataset2_clean.iloc[:,3])
col_11 = dc.normalize(dataset2_clean.iloc[:,11])
col_13 = dc.normalize(dataset2_clean.iloc[:,13])

dataset2_clean.iloc[:,1] = col_1
dataset2_clean.iloc[:,3] = col_3
dataset2_clean.iloc[:,11] = col_11
dataset2_clean.iloc[:,13] = col_13


#-----------------------------DATASET3 MODIFICATION------------------------------
# - change 7th feature's datatype from str to int64
dataset3_clean.astype({6: 'int64'})

# - drop 1st feature, which is the sample ID name
dataset3_clean = dataset3_clean.drop(columns = {dataset3_clean.columns[1]})

codes, uniques = pd.factorize(dataset3_clean.iloc[:,dataset3_clean.columns[-1]])
            
# change labels to 0 and 1
dataset3_clean.iloc[:,dataset3_clean.columns[-1]] = codes



#-----------------------------DATASET4 MODIFICATION------------------------------
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


# normalize columns 2,3,4,5,6,7
col_1 = dc.normalize(dataset4_clean.iloc[:,1])
col_2 = dc.normalize(dataset4_clean.iloc[:,2])
col_3 = dc.normalize(dataset4_clean.iloc[:,3])
col_4 = dc.normalize(dataset4_clean.iloc[:,4])
col_5 = dc.normalize(dataset4_clean.iloc[:,5])
col_6 = dc.normalize(dataset4_clean.iloc[:,6])

dataset4_clean.iloc[:,1] = col_1
dataset4_clean.iloc[:,2] = col_2
dataset4_clean.iloc[:,3] = col_3
dataset4_clean.iloc[:,4] = col_4
dataset4_clean.iloc[:,5] = col_5
dataset4_clean.iloc[:,6] = col_6



#--------------------------------------------------------------------------------
# compute and output stats for each dataset
stats_1 = dc.data_stats(dataset1_clean)
stats_2 = dc.data_stats(dataset2_clean)
stats_3 = dc.data_stats(dataset3_clean)
stats_4 = dc.data_stats(dataset4_clean)



#%%
#------DATA CLEANING for NAIVE BAYES (are numpy arrays after data_prep_naive())--------
# does not contain extra column of 1s, features and labels are put together
dataset1_arr_naive = dataset1_clean.drop(dataset1_clean.columns[0], axis=1).to_numpy()
dataset2_arr_naive = dc.data_prep_naive(dataset2_clean)
dataset2_arr_test_naive = dc.data_prep_naive(dataset2_clean_test)
dataset3_arr_naive = dc.data_prep_naive(dataset3_clean)
dataset4_arr_naive = dataset4_clean.drop(dataset4_clean.columns[0], axis=1).to_numpy()


#%%
#------------------------------TEST LOGISTIC REGRESSION-----------------------------
cvo = cv.Cross_Validation()

#test single dataset
log_score, sk_log_score = cvo.log_k_fold(5, dataset1_clean, 0.1, 0.01, 0)
print('\n', 'Logistic Regression Cross Validation Evaluation: ', log_score)
#log_score, sk_log_score = cvo.log_k_fold_iter(5, dataset3_clean, 0.1, 1500, 2)

# test various learning rates alpha
#a = np.array([0.0001, 0.01, 0.1, 2, 5])
#log_score, sk_log_score, time_array, avg_acc, avg_prec, avg_recall = test_diff_a(a, dataset1_clean)


# test various number of iterations
#num_iter = np.array([100,1000,5000,10000,15000])
#log_score, sk_log_score = test_diff_iter(num_iter, dataset2_clean)


#------------------------------Plotting functions -----------------------------

#plt.plot(a, time_array)
#plt.title('Learning Rate v.s. Convergence Time (Dataset 1)')
#plt.xlabel('learning rate(a)')
#plt.ylabel('convergence time(t)')
#plt.show()
#
#ax = plt.subplot(111)
#ax.plot(a, avg_acc, label='avg_accuracy')
#ax.plot(a, avg_prec, label='avg_precision')
#ax.plot(a, avg_recall, label='avg_recall')
#plt.title('Learning Rate v.s. Accuracy/Precision/Recall (Dataset 1)')
#plt.legend()
#plt.xlabel('learning rate(a)')
#plt.ylabel('Accuracy/Precision/Recall')
#plt.show()

#%%
#----------------------------TEST NAIVE BAYES----------------------------
nb_score, sk_nb_score = cvo.naive_k_fold(5, dataset1_arr_naive)
print('\n','Naive Bayes Cross Validation Evaluation: ', nb_score)
#nb_score_nb, sk_nb_score = cvo.naive_k_fold(5, dataset1_arr_naive)
#nb_score_nb, sk_nb_score = cvo.naive_k_fold(5, dataset2_arr_naive)
#nb_score_nb, sk_nb_score = cvo.naive_k_fold(5, dataset3_arr_naive)


