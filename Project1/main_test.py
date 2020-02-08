#%%
import pandas as pd
import numpy as np
import Data_Cleaner as cd
import Log_Regression as lgr
import Naive_Bayes as nb
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


#root_path = '/Users/liuxijun/Downloads/Applied_Machine_Learning/Project1/'
#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project1/'
root_path = '/Users/kirenrao/Documents/GitHub/Applied_Machine_Learning/Project1/'


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

#------------------------further modification of DATASET3------------------------
# - change 7th feature's datatype from str to int64
dataset3_clean.astype({6: 'int64'})

# - drop 1st feature, which is the sample ID name
dataset3_clean = dataset3_clean.drop(dataset3_clean.columns[0], axis=1)


#------------------------further modifications of DATASET4-----------------------
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



#%%
#------DATA CLEANING for NAIVE BAYES (are numpy arrays after data_prep_naive())--------
# does not contain extra column of 1s, features and labels are put together
dataset1_arr_naive = dc.data_prep_naive(dataset1_clean)
dataset2_arr_naive = dc.data_prep_naive(dataset2_clean)
dataset2_arr_test_naive = dc.data_prep_naive(dataset2_clean_test)
dataset3_arr_naive = dc.data_prep_naive(dataset3_clean)
dataset4_arr_naive = dc.data_prep_naive(dataset4_clean)
#print(dataset1_arr_naive[0:20,:])
#print(dataset1_arr_naive.shape)



#%%
# 1. extract labels (y) from dataset for LOG REG
# 2. delete labels column from dataset (keeps all features)
y1, dataset1_log = split_y(dataset1_clean)
y2, dataset2_log = split_y(dataset2_clean)
y2_test, dataset2_test_log = split_y(dataset2_clean_test)
y3, dataset3_log = split_y(dataset3_clean)
y4, dataset4_log = split_y(dataset4_clean)


# change all LOG REG datasets to numpy arrays
dataset1_arr = dataset1_log.to_numpy()
dataset2_arr = dataset1_log.to_numpy()
dataset2_arr_test = dataset2_clean_test.to_numpy()
dataset3_arr = dataset1_clean.to_numpy()
dataset4_arr = dataset1_clean.to_numpy()



#%%
#--------------------------PREPARE TEST DATA--------------------------------
# Add 1 to X in the front
#shape0, shape1 = dataset1_arr.shape
#train_data = dataset1_arr[np.arange(int(shape0/5*4))]
#test_data = dataset1_arr[np.arange(int(shape0/5*4),shape0)]
#train_y = y1[np.arange(int(shape0/5*4))]
#test_y = y1[np.arange(int(shape0/5*4), shape0)]

shape0, shape1 = dataset2_arr.shape
train_data = dataset2_arr[np.arange(int(shape0/5*4))]
test_data = dataset2_arr[np.arange(int(shape0/5*4),shape0)]
train_y = y2[np.arange(int(shape0/5*4))]
test_y = y2[np.arange(int(shape0/5*4), shape0)]

#%%
#--------------------TEST LOGISTIC REGRESSION-----------------
N,m = train_data.shape

# OUR MODEL
lg = lgr.Log_Regression(np.zeros(m))
lg.fit(train_data, train_y, 0.1, 0.01)
y_pred = lg.predict(test_data)
accuracy = lg.evaluate(test_y, y_pred)

# SK-LEARN
clf = LogisticRegression()
clf.fit(train_data, train_y)
y_pred_ski = clf.predict(test_data)
accuracy_ski = clf.score(test_data, test_y)

print(accuracy)
print(accuracy_ski)







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

# OUR MODEL
nbc = nb.Naive_Bayes()


summaries = nbc.fit(dataset1_arr_naive)
print("sum,ary",summaries)
totalRows = dataset1_arr_naive.shape[0]
print("total row",totalRows)
print("test_y", test_y.shape, "type",type(test_y))
predicted = np.array([])
lable = np.array([])

for i in range(totalRows):
    prediction = nbc.predict(summaries, dataset1_arr_naive[i], totalRows)
    #print(dataset1_arr_naive[i][-1])
    #print(max(prediction.items(), key=operator.itemgetter(1))[0])
    predicted = np.append(predicted, max(prediction.items(), key=operator.itemgetter(1))[0])
    lable = np.append(lable, dataset1_arr_naive[i][-1])
    #comparison = nbc.evaluate(lable, predicted)
    #print("comparison",comparison)
print(predicted.shape,lable.shape)
comparison = nbc.evaluate(lable, predicted)
print("accuracy",comparison)
    
# SKLEARN
#clf_nb = GaussianNB()
#clf_nb.fit(train_data, train_y)
#y_pred_ski_nb = clf.predict(test_data)
#accuracy_ski_nb = clf.score(test_data, test_y)

#print(accuracy_ski_nb)










