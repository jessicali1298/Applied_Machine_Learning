#%%
import pandas as pd
import numpy as np
import Data_Cleaner as cd
import Log_Regression as lgr
import Naive_Bayes as nb
import Cross_Validation as cv
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



#root_path = '/Users/liuxijun/Downloads/Applied_Machine_Learning/Project1/'
<<<<<<< HEAD
#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
root_path = '/Users/kirenrao/Documents/GitHub/Applied_Machine_Learning/Project1/'
=======
root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
#root_path = '/Users/kirenrao/Documents/GitHub/Applied_Machine_Learning/Project1/'
>>>>>>> dc964e57c48095b31c9e93b46a0c0c97b7b487bb


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
    
    for j in range(a_arr.shape[0]):
        log_acc, sk_log_acc = cvo.log_k_fold(5,input_data,a_arr[j],0.01)
        log_score_ls.append(log_acc)
        sk_log_score_ls.append(sk_log_acc)
        
    return log_score_ls, sk_log_score_ls

def test_diff_iter(iter_arr, input_data):
    log_score_ls = []
    sk_log_score_ls = []
    
    for j in range(iter_arr.shape[0]):
        log_acc, sk_log_acc = cvo.log_k_fold_iter(5, input_data, 0.1, iter_arr[j])
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
<<<<<<< HEAD
<<<<<<< HEAD
#--------------------------PREPARE TEST DATA--------------------------------
# Add 1 to X in the front
shape0, shape1 = dataset1_arr.shape
train_data = dataset1_arr[np.arange(int(shape0/5*4))]
test_data = dataset1_arr[np.arange(int(shape0/5*4),shape0)]
train_y = y1[np.arange(int(shape0/5*4))]
test_y = y1[np.arange(int(shape0/5*4), shape0)]

#shape0, shape1 = dataset2_arr.shape
#train_data = dataset2_arr[np.arange(int(shape0/5*4))]
#test_data = dataset2_arr[np.arange(int(shape0/5*4),shape0)]
#train_y = y2[np.arange(int(shape0/5*4))]
#test_y = y2[np.arange(int(shape0/5*4), shape0)]

#%%
#--------------------TEST LOGISTIC REGRESSION-----------------
N,m = train_data.shape
#
## OUR MODEL
lg = lgr.Log_Regression(np.zeros(m))
lg.fit(train_data, train_y, 0.1, 0.01)
y_pred = lg.predict(test_data)
accuracy = lg.evaluate(test_y, y_pred)
print(accuracy)

#lg = lgr.Log_Regression(np.zeros(m))


#splited_log = cv.Cross_Validation.dataset_sep(dataset1_arr, 5)
#result_log = cv.Cross_Validation.k_fold_log(splited_log, 5)


#print("\n" + "result: ", result_log)
=======
=======
>>>>>>> dc964e57c48095b31c9e93b46a0c0c97b7b487bb
#------DATA CLEANING for NAIVE BAYES (are numpy arrays after data_prep_naive())--------
# does not contain extra column of 1s, features and labels are put together
dataset1_arr_naive = dataset1_clean.drop(dataset1_clean.columns[0], axis=1).to_numpy()
dataset2_arr_naive = dc.data_prep_naive(dataset2_clean)
dataset2_arr_test_naive = dc.data_prep_naive(dataset2_clean_test)
dataset3_arr_naive = dc.data_prep_naive(dataset3_clean)
dataset4_arr_naive = dataset4_clean.drop(dataset4_clean.columns[0], axis=1).to_numpy()

<<<<<<< HEAD
>>>>>>> c24273cf9ba4758ff3487f9ad6ffd0261c8cb4c0
=======
>>>>>>> dc964e57c48095b31c9e93b46a0c0c97b7b487bb

#%%
#------------------------------TEST LOGISTIC REGRESSION-----------------------------
cvo = cv.Cross_Validation()

<<<<<<< HEAD
<<<<<<< HEAD
#print(accuracy)
print(accuracy_ski)
=======
#test single dataset
log_score, sk_log_score = cvo.log_k_fold(5, dataset1_clean, 0.1, 0.01)
#log_score, sk_log_score = cvo.log_k_fold_iter(5, dataset1_clean, 0.1, 1000)
>>>>>>> c24273cf9ba4758ff3487f9ad6ffd0261c8cb4c0
=======
#test single dataset
#log_score, sk_log_score = cvo.log_k_fold(5, dataset1_clean, 0.1, 0.01, 0)
log_score, sk_log_score = cvo.log_k_fold_iter(5, dataset1_clean, 0.1, 1500, 2.3)
>>>>>>> dc964e57c48095b31c9e93b46a0c0c97b7b487bb

# test various learning rates alpha
#a = np.array([0.0001, 0.01, 0.1, 2, 5])
#log_score, sk_log_score = test_diff_a(a, dataset4_clean)

# test various number of iterations
#num_iter = np.array([100,1000,5000,10000,15000])
#log_score, sk_log_score = test_diff_iter(num_iter, dataset2_clean)





#%%
#----------------------------TEST NAIVE BAYES----------------------------
# Test calculating class probabilities

dataset = np.array([[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]])



#shape0, shape1 = dataset2_arr_naive.shape
#train_data_nb = dataset2_arr_test_naive[np.arange(int(shape0/5*4))]
#test_data_nb = dataset2_arr_test_naive[np.arange(int(shape0/5*4),shape0)]
#train_y_nb = y2[np.arange(int(shape0/5*4))]
#test_y_nb = y2[np.arange(int(shape0/5*4), shape0)]



#trainDataWithLable = np.append(train_data_nb,train_y_nb.reshape(train_data_nb.shape[0],1),axis=1)
#print(trainDataWithLable)
#
#testDataWithLable = np.append(test_data_nb,test_y_nb.reshape(test_data_nb.shape[0],1),axis=1)
##print(testDataWithLable)
import operator
    

# OUR MODEL
nbc = nb.Naive_Bayes()


splited_naive = np.array_split(dataset1_arr_naive, 5)
<<<<<<< HEAD
testDataWithLable = splited_naive[0]
trainDataWithLable = np.concatenate(np.delete(splited_naive,0,0),axis=0)
=======
testDataWithLabel = splited_naive[0]
trainDataWithLabel = np.concatenate(np.delete(splited_naive,0,0),axis=0)
>>>>>>> dc964e57c48095b31c9e93b46a0c0c97b7b487bb



summaries = nbc.fit(trainDataWithLabel)
print("summaries",summaries)
totalRows = trainDataWithLabel.shape[0]
print("total row",totalRows)
#print("test_y", test_y.shape, "type",type(test_y))
predicted = np.array([])
lable = np.array([])

for i in range(testDataWithLabel.shape[0]):
    prediction = nbc.predict(summaries, testDataWithLabel[i], totalRows)
    predicted = np.append(predicted, max(prediction.items(), key=operator.itemgetter(1))[0])
    lable = np.append(lable, testDataWithLabel[i][-1])
comparison = nbc.evaluate(lable, predicted)
print("accuracy of implementation",comparison)
 
# SKLEARN
# split data for SKlearn's inputs
y1, sk_data1 = cvo.split_y(dataset4_clean)
sk_data1_arr = sk_data1.to_numpy()

splited_y = np.array_split(y1, 5)
splited_data = np.array_split(sk_data1_arr,5)

test_y = splited_y[0]
train_y = np.concatenate(np.delete(splited_y,0,0),axis=0)

test_data = splited_data[0]
train_data = np.concatenate(np.delete(splited_data,0,0), axis=0)

#test SKLEARN
clf_nb = GaussianNB()
clf_nb.fit(train_data, train_y)
y_pred_ski_nb = clf_nb.predict(test_data)
accuracy_ski_nb = clf_nb.score(test_data, test_y)

print(accuracy_ski_nb)







