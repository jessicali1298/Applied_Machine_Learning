import numpy as np

class Log_Regression:
    
    def __init__(self, weight):
        self.weight = weight


#%%
# Logistic Regression:
# 1. Use gradient descent to find weights for log-odds ratio
# 2. Compute predicted probability (sigmoid function)
# 3 prob > 0.5, y = 1;   prob < 0.5, y = 0;   prob = 0.5, use random decider

    #------------------PSEUDO-CODE-------------------
    def gradient(self,X,y,w):
        N,m = X.shape
    #    th = 1/(1+np.exp(-np.dot(w.T, X)))
        th = self.logistic(X, w.T)
    #    th = logistic(np.dot(X,w))
        grad = (1/N) * np.dot(X.T, th-y)
        return grad
    
    def gradientDescent(self, X, y, a, end_cond):
        N,m = X.shape
        w = np.zeros(m)
        
        g = np.inf
        
        while np.linalg.norm(g) > end_cond:
            g = self.gradient(X,y,w)
            w = w - a*g
        return w
    
    
    def logistic(self,input1, input2):
            ans = 1/(1+np.exp(-np.dot(input1, input2)))
            return ans
    
        
    def fit(self, X, y, a, end_cond):
        #CODE
        # N = number of instances, m = number of features
        N,m = X.shape
        
        w = self.gradientDescent(X, y, a, end_cond)
        self.weight = w
    
    def predict(self, X):
        #CODE
        N,m = X.shape
        w = self.weight
        prob = self.logistic(X, w.T)
        
        class_1_idx = np.where(prob > 0.5)[0]
        class_0_idx = np.where(prob < 0.5)[0]
        equal_prob_idx = np.where(prob == 0.5)[0]
        
        pred_arr = np.empty(N)
        
        pred_arr[class_1_idx] = 1
        pred_arr[class_0_idx] = 0
        pred_arr[equal_prob_idx] = np.random.choice(2,1)
            
        return pred_arr
        
<<<<<<< HEAD
    def evaluate(y_truth, y_pred):
        comparison = np.equal(y_truth, y_pred)
        accuracy = (np.where(comparison == True)[0].size)/comparison.size
=======
    def evaluate(self, y_truth, y_pred):
        comparison = np.equal(y_truth, y_pred)
        accuracy = (np.where(comparison == True)[0].size)/comparison.size
        
>>>>>>> 826c9f98e17b5e51e0efa1f85cbd8becab22d300
        return accuracy
       
        
        