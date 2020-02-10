import numpy as np
import Data_Cleaner as dc
import Log_Regression as lgr
from sklearn.linear_model import LogisticRegression


class Cross_Validation: 
    
    def split_y(self, input_data):
        dco = dc.Data_Cleaner()
        y = dco.extract_y(input_data).to_numpy()
        output_data = input_data.drop(columns = {input_data.columns[-1]})
        return y, output_data


    def score(self, test_label, pred_label):
        TP = (np.where((test_label == 1) & (pred_label == 1))[0]).shape[0]
        FP = (np.where((test_label == 1) & (pred_label == 0))[0]).shape[0]
        TN = (np.where((test_label == 0) & (pred_label == 0))[0]).shape[0]
        FN = (np.where((test_label == 0) & (pred_label == 1))[0]).shape[0]
        
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        
        if (TP+FP == 0):
            precision = TP/0.00001
        else:
            precision = TP/(TP+FP)
            
        if(TP+FN == 0):
            recall = TP/0.00001
        else:
            recall = TP/(TP+FN)
        
        return accuracy, precision, recall
        
        
        
        
        
    def log_k_fold(self, k, dataset, a, epsilon):
        # shuffle the dataset
        dataset_shuffle = dataset.sample(frac=1).reset_index(drop = True)
        
        # separate labels from features
        y, dataset_log = self.split_y(dataset_shuffle)
        dataset_arr = dataset_log.to_numpy()
        
        # split dataset into k folds
        splited_sample = np.array_split(dataset_arr, k)
        splited_y = np.array_split(y, k)
        
        # begin evaluation
        accuracy_ls = []
        precision_ls = []
        recall_ls = []
        accuracy_ls_sk = []
        precision_ls_sk = []
        recall_ls_sk = []
        
        for i in range(k):
            # copy the datasets so the original data will not be messed up
            copy_sample = np.copy(splited_sample)
            copy_y = np.copy(splited_y)
            
            # define test data and train data
            test_data = copy_sample[i]
            test_y = copy_y[i]
                
            train_data = np.concatenate(np.delete(copy_sample,i,0), axis=0)
            train_y = np.concatenate(np.delete(copy_y,i,0), axis=0)
            
            # OUR MODEL
            N,m = train_data.shape
        
            lg = lgr.Log_Regression(np.zeros(m))
            lg.fit(train_data, train_y, a, epsilon)
            y_pred = lg.predict(test_data)
            accuracy, precision, recall = self.score(test_y, y_pred)

            accuracy_ls.append(accuracy)
            precision_ls.append(precision)
            recall_ls.append(recall)
        
            # SK-LEARN
            clf = LogisticRegression()
            clf.fit(train_data, train_y)
            y_pred_sk = clf.predict(test_data)
            print(clf.score(test_data, test_y))
            accuracy_sk, precision_sk, recall_sk = self.score(test_y, y_pred_sk)
            
            accuracy_ls_sk.append(accuracy_sk)
            precision_ls_sk.append(precision_sk)
            recall_ls_sk.append(recall_sk)
            
            # create two score dictionaries
            score_log = {'accuracy': accuracy_ls, 'precision': precision_ls, 'recall': recall_ls} 
            score_sk = {'accuracy': accuracy_ls_sk, 'precision': precision_ls_sk, 'recall': recall_ls_sk} 
            
        return score_log, score_sk
    
    def log_k_fold_iter(self,k, dataset, a, num_iter):
        # shuffle the dataset
        dataset_shuffle = dataset.sample(frac=1).reset_index(drop = True)
        
        # separate labels from features
        y, dataset_log = self.split_y(dataset_shuffle)
        dataset_arr = dataset_log.to_numpy()
        
        # split dataset into k folds
        splited_sample = np.array_split(dataset_arr, k)
        splited_y = np.array_split(y, k)
        
        # begin evaluation
        accuracy_ls = []
        precision_ls = []
        recall_ls = []
        accuracy_ls_sk = []
        precision_ls_sk = []
        recall_ls_sk = []
        
        for i in range(k):
            # copy the datasets so the original data will not be messed up
            copy_sample = np.copy(splited_sample)
            copy_y = np.copy(splited_y)
            
            # define test data and train data
            test_data = copy_sample[i]
            test_y = copy_y[i]
                
            train_data = np.concatenate(np.delete(copy_sample,i,0), axis=0)
            train_y = np.concatenate(np.delete(copy_y,i,0), axis=0)
            
            # OUR MODEL
            N,m = train_data.shape
        
            lg = lgr.Log_Regression(np.zeros(m))
            lg.fit_iter(train_data, train_y, a, num_iter)
            y_pred = lg.predict(test_data)
            accuracy, precision, recall = self.score(test_y, y_pred)

            accuracy_ls.append(accuracy)
            precision_ls.append(precision)
            recall_ls.append(recall)
        
            # SK-LEARN
            clf = LogisticRegression()
            clf.fit(train_data, train_y)
            y_pred_sk = clf.predict(test_data)
            accuracy_sk, precision_sk, recall_sk = self.score(test_y, y_pred_sk)
            
            accuracy_ls_sk.append(accuracy_sk)
            precision_ls_sk.append(precision_sk)
            recall_ls_sk.append(recall_sk)
            
            # create two score dictionaries
            score_log = {'accuracy': accuracy_ls, 'precision': precision_ls, 'recall': recall_ls} 
            score_sk = {'accuracy': accuracy_ls_sk, 'precision': precision_ls_sk, 'recall': recall_ls_sk} 
            
        return score_log, score_sk
    
    
#%%  
#    def dataset_sep(self,dataset,k):
#        rand_data = np.take(dataset,np.random.permutation(dataset.shape[0]),axis=0)             # shuffle the dataset
#        sep_dataset = np.array_split(rand_data, k)            # split the shuffled dataset into k piles
#        
#        temp = []
#        mer = []
#        result = []
#        for i in range(k):
#            for n in range(k):
#                if n!= i:
#                    mer.append(sep_dataset[n])
#            mer = np.concatenate((mer), axis=0)
#            temp.append(mer)
#            temp.append(sep_dataset[i])
#            result.append(np.array(temp))
#            mer = []
#            temp = []
#        return result
#    # The result is an array with length k, of which the elements are organized
#    # in the order of [[training data], [testing data]]
#    # So, the data structure of 'result' of k = 5 is
#    # [[[training data 1], [testing data 1]],
#    #  [[training data 2], [testing data 2]],
#    #  [[training data 3], [testing data 3]],
#    #  [[training data 4], [testing data 4]],
#    #  [[training data 5], [testing data 5]]]
#    
#    def split_y_array(self,input_data):
#        y = []
#        output_data =[]
#        for i in range(len(input_data)):
#            y.append(input_data[i][-1])
#            output_data.append(input_data[i][0:len(input_data)])
#        
#        return np.asarray(y), np.asarray(output_data)
#    
#    def k_fold_log(self,dataset, k):
#        accuracy = 0
#        recall = 0
#        precision = 0
#        true_pos = 0
#        true_neg = 0
#        false_pos = 0
#        false_neg = 0
#        result = []
#        for i in range(k):
#            train_label, train_data = self.split_y_array(dataset[i][0])
#            test_label, test_data = self.split_y_array(dataset[i][1])
#            
#            N,m = train_data.shape
#            lg = lgr.Log_Regression(np.zeros(m))
#            lg.fit(train_data, train_label, 0.1, 0.01)
#            pred_label = lg.predict(test_data)
#            for n in range (len(test_label)):
#                if test_label == 1  and pred_label == 1:
#                    true_pos = true_pos + 1
#                if test_label == 1 and pred_label == 0:
#                    false_pos = false_pos + 1
#                if test_label == 0 and pred_label == 0:
#                    true_neg = true_neg + 1
#                if test_label == 0 and pred_label == 1:
#                    false_neg = false_neg + 1
#            accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
#            precision = true_pos / (true_pos + false_pos)
#            recall = true_pos / (true_pos + false_neg)
#            result.append(['accuracy'+str(i)+': ', accuracy, 'precision'+str(i)+': ', precision, 'recall'++str(i)+': ', recall])
#            accuracy = 0
#            recall = 0
#            precision = 0
#            true_pos = 0
#            true_neg = 0
#            false_pos = 0
#            false_neg = 0
#        return result
#    
#            

