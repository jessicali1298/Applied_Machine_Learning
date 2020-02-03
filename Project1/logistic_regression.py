import numpy as np
import matplotlib as plt
import pandas as pd
import clean_data_class as cd

#%%
dc = cd.DataCleaner()

dataset1_clean = dc.data_prep(dataset1, 'DATASET1--IONOSPHERE')

#%%
# Logistic Regression:
# 1. Use gradient descent to find weights for log-odds ratio
# 2. Compute predicted probability (sigmoid function)
# 3 prob > 0.5, y = 1;   prob < 0.5, y = 0;   prob = 0.5, use random decider


#------------------PSEUDO-CODE-------------------
def gradient(X,y,w):
    N,D = X.shape
    th = 1/(1+np.exp(-np.dot(w.T, X)))
#    th = logistic(np.dot(X,w))
    grad = (1/N) * np.dot(X.T, th-y)
    return grad

def gradientDescent(X, y, a, end_cond):
    N,D = X.shape
    w = np.zeros(D)
    
    g = np.inf
    
    while np.linalg.norm(g) > end_cond:
        g = gradient(X,y,w)
        w = w - a*g
    return w


# remember to add 1 to X
    
def fit(X, y, a, end_cond):
    #CODE
    # N = number of instances, D = number of features
    N,D = X.shape
    
    w = gradientDescent(X, y, a, end_cond)
    prob = 1/(1+np.exp(-np.dot(w.T, X)))
    
    class_1_idx = np.where(prob > 0.5)[0]
    class_0_idx = np.where(prob < 0.5)[0]
    equal_prob_idx = np.where(prob == 0.5)[0]
    
    pred_arr = np.empty(N)
    
    pred_arr[class_1_idx] = 1
    pred_arr[class_0_idx] = 0
    pred_arr[equal_prob_idx] = np.random.choice(2,1)
        
    return pred_arr

def predict(X, a, end_cond):
    #CODE
    prob = 1/(1+np.exp(-np.dot(w.T, X)))
    
def evaluate(y_truth, y_pred):
    #CODE
    comparison = np.equal(y_truth, y_pred)
    accuracy = (np.where(comparison == False)[0].size)/comparison.size
    
    return accuracy
   
    
#%%
    
#------------------------TESTING-----------------------

    
#%%

#RANDOM TESTING    
    
arr = np.array([[1,2,3,],[4,5,6]])
print(arr.shape)
arr[[0,1]] = [8,8]
print(arr[[0,1,2,3]])
print(arr.shape)
print(1+np.exp([0,1,2]))   

for i in range(8): 
    print(np.random.choice(2,1))
    
    