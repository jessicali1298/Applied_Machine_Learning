import pandas as pd
import numpy as np
import Data_Cleaner as cd
import Log_Regression as lgr
from sklearn.linear_model import LogisticRegression



class Cross_Validation: 
    
    def dataset_sep(dataset, k):
        rand_data = np.take(dataset,np.random.permutation(dataset.shape[0]),axis=0)             # shuffle the dataset
        sep_dataset = np.array_split(rand_data, k)            # split the shuffled dataset into k piles
        temp = []
        mer = []
        result = []
        for i in range(k):
            for n in range(k):
                if n!= i:
                    mer.append(sep_dataset[n])
            mer = np.concatenate((mer), axis=0)
            temp.append(mer)
            temp.append(sep_dataset[i])
            result.append(np.array(temp))
            mer = []
            temp = []
        return result
    # The result is an array with length k, of which the elements are organized
    # in the order of [[training data], [testing data]]
    # So, the data structure of 'result' of k = 5 is
    # [[[training data 1], [testing data 1]],
    #  [[training data 2], [testing data 2]],
    #  [[training data 3], [testing data 3]],
    #  [[training data 4], [testing data 4]],
    #  [[training data 5], [testing data 5]]]
    
    def split_y_array(input_data):
        y = []
        output_data =[]
        for i in range(len(input_data)):
            y.append(input_data[i][-1])
            output_data.append(input_data[i][:-1])
        
        return np.array(y), np.array(output_data)
    
    def k_fold_log(dataset, k):
        accuracy = 0
        recall = 0
        precision = 0
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        result = []
        for i in range(k):
            train_label, train_data = Cross_Validation.split_y_array(dataset[i][0])
            test_label, test_data = Cross_Validation.split_y_array(dataset[i][0])
            N,m = train_data.shape
            lg = lgr.Log_Regression(np.zeros(m))
            lg.fit(train_data, train_label, 0.1, 0.01)
            pred_label = lg.predict(test_data)
            for n in range (len(test_label)):
                if test_label == 1  and pred_label == 1:
                    true_pos = true_pos + 1
                if test_label == 1 and pred_label == 0:
                    false_pos = false_pos + 1
                if test_label == 0 and pred_label == 0:
                    true_neg = true_neg + 1
                if test_label == 0 and pred_label == 1:
                    false_neg = false_neg + 1
            accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            result.append(['accuracy'+str(i)+': ', accuracy, 'precision'+str(i)+': ', precision, 'recall'++str(i)+': ', recall])
            accuracy = 0
            recall = 0
            precision = 0
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
        return result
    
            
#%%
