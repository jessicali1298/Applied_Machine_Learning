import numpy as np
import Data_Cleaner as dc
import Log_Regression as lgr
import Naive_Bayes as nb
from sklearn.linear_model import LogisticRegression
import operator
from sklearn.naive_bayes import GaussianNB


class Cross_Validation: 
    
    def split_y(self, input_data):
        dco = dc.Data_Cleaner()
        y = dco.extract_y(input_data).to_numpy()
        output_data = input_data.drop(columns = {input_data.columns[-1]})
        return y, output_data


    def score(self, test_label, pred_label):
        accuracy = 0
        recall = 0
        precision = 0
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        test_y = 0
        pred_y = 0
        for n in range (len(test_label)):
            test_y = test_label[n]
            pred_y = pred_label[n]
            if test_y == 1  and pred_y == 1:
                true_pos = true_pos + 1
            if test_y == 1 and pred_y == 0:
                false_pos = false_pos + 1
            if test_y == 0 and pred_y == 0:
                true_neg = true_neg + 1
            if test_y == 0 and pred_y == 1:
                false_neg = false_neg + 1
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        
        return accuracy, precision, recall
        
        
        
        
        
    def log_k_fold(self, k, dataset, a, epsilon, lamda):
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
            lg.fit(train_data, train_y, a, epsilon, lamda)
            y_pred = lg.predict(test_data)
            accuracy, precision, recall = self.score(test_y, y_pred)

            accuracy_ls.append(accuracy)
            precision_ls.append(precision)
            recall_ls.append(recall)
        
            # SK-LEARN
            clf = LogisticRegression()
            clf.fit(train_data, train_y)
            y_pred_sk = clf.predict(test_data)
            print(i, ':', clf.score(test_data, test_y))
            accuracy_sk, precision_sk, recall_sk = self.score(test_y, y_pred_sk)
            
            accuracy_ls_sk.append(accuracy_sk)
            precision_ls_sk.append(precision_sk)
            recall_ls_sk.append(recall_sk)
            
            # create two score dictionaries
            score_log = {'accuracy': accuracy_ls, 'precision': precision_ls, 'recall': recall_ls} 
            score_sk = {'accuracy': accuracy_ls_sk, 'precision': precision_ls_sk, 'recall': recall_ls_sk} 
            
        return score_log, score_sk
    
    def log_k_fold_iter(self,k, dataset, a, num_iter, lamda):
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
            lg.fit_iter(train_data, train_y, a, num_iter, lamda)
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
    
    def naive_k_fold(self, k, dataset):
        # shuffle the dataset
        dataset_shuffle = np.take(dataset,np.random.permutation(dataset.shape[0]),axis=0) 
        
        
        # split dataset into k folds
        splited_sample = np.array_split(dataset_shuffle, k)
        
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

            # define test data and train data
            test_data = copy_sample[i]
            test_y = []
            for x in range(len(test_data)):
                test_y.append(test_data[x][-1])
            

            train_data = np.concatenate(np.delete(copy_sample,i,0), axis=0)
            train_y = []
            for x in range(len(train_data)):
                train_y.append(train_data[x][-1])
            
            # OUR MODEL
            totalRows,m = train_data.shape
        
            nbc = nb.Naive_Bayes()
            summaries = nbc.fit(train_data)
            predicted = np.array([])
            label = np.array([])

            for x in range(test_data.shape[0]):
               prediction = nbc.predict(summaries, test_data[x], totalRows)
               predicted = np.append(predicted, max(prediction.items(), key=operator.itemgetter(1))[0])
               label = np.append(label, test_data[x][-1])
            accuracy, precision, recall = self.score(label, predicted)
            
            

            accuracy_ls.append(accuracy)
            precision_ls.append(precision)
            recall_ls.append(recall)

            # SK-LEARN
            nbc = nb.Naive_Bayes()
            clf_nb = GaussianNB()
            clf_nb.fit(train_data, train_y)
            y_pred_sk = clf_nb.predict(test_data)
            accuracy_ski_nb = clf_nb.score(test_data, test_y)
            accuracy_sk, precision_sk, recall_sk = self.score(test_y, y_pred_sk)
            
            
            
            accuracy_ls_sk.append(accuracy_sk)
            precision_ls_sk.append(precision_sk)
            recall_ls_sk.append(recall_sk)
            

            score_log = {'accuracy': accuracy_ls, 'precision': precision_ls, 'recall': recall_ls} 
            score_sk = {'accuracy': accuracy_ls_sk, 'precision': precision_ls_sk, 'recall': recall_ls_sk} 
            
        return score_log, score_sk
    

