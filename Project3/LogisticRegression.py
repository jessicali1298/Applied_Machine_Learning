import numpy as np

class LogisticRegression:
    
    def __init__(self, weight):
        self.weight = weight
#%%
# Logistic Regression:
# 1. Use gradient descent to find weights for log-odds ratio
# 2. Compute predicted probability (sigmoid function)
# 3 prob > 0.5, y = 1;   prob < 0.5, y = 0;   prob = 0.5, use random decider
    def logistic(self,input1, input2):
        z = np.dot(input1, input2)
        
        pos_num = np.where(z>=0)[0]
        neg_num = np.where(z<0)[0]
        
        ans = np.empty(z.shape[0])
        
        ans[pos_num] = 1/(1+np.exp(-z[pos_num]))
        ans[neg_num] = np.exp(z[neg_num])/(1+np.exp(z[neg_num]))
    
        return ans
    
    def gradient(self,X,y,w):
        N,m = X.shape
    #    th = 1/(1+np.exp(-np.dot(w.T, X)))
        th = self.logistic(X, w.T)
#        print("sigmoid: ", th)
    #    th = logistic(np.dot(X,w))
        grad = (1/N) * np.dot(X.T, th-y)
        return grad
    
    
    # using epsilon as stopping criteria
    def gradientDescent(self, X, y, a, end_cond, lamda):

        N,m = X.shape
        w = np.zeros(m)
        
        g = np.inf
        
        while np.linalg.norm(g) > end_cond:
            g = self.gradient(X,y,w)
#            w = w - a*g
            w = w - a*(g + (1/N)*lamda*w)
        return w
    
    # using num. of iterations as stopping criteria
    def gradientDescent_iter(self, X, y, a, num_iter, lamda):

        N,m = X.shape
        w = np.zeros(m)
        
        g = np.inf
        i = 0
        while i < num_iter:
            g = self.gradient(X,y,w)

            w = w - a*(g + (1/N)*lamda*w)

            i = i+1
        return w
    

    # fit using epsilon as stopping criteria for GD
    def fit(self, X, y, a, end_cond, lamda):
        # N = number of instances, m = number of features
        N,m = X.shape
        
        w = self.gradientDescent(X, y, a, end_cond, lamda)
        self.weight = w
    
    
    # fit using number of iterations as stopping criteria for gradient descent
    def fit_iter(self, X, y, a, num_iter, lamda):
        N,m = X.shape
        
        w = self.gradientDescent_iter(X, y, a, num_iter, lamda)
        self.weight = w
        
    def predict(self, X):
        #CODE
        N,m = X.shape
        w = self.weight
        prob = self.logistic(X, w.T)
        
#        class_1_idx = np.where(prob > 0.5)[0]
#        class_0_idx = np.where(prob < 0.5)[0]
#        equal_prob_idx = np.where(prob == 0.5)[0]
#        
#        pred_arr = np.empty(N)
#        
#        pred_arr[class_1_idx] = 1
#        pred_arr[class_0_idx] = 0
#        pred_arr[equal_prob_idx] = np.random.choice(2,1)
            
        return prob
        

    def evaluate(self, y_truth, y_pred):
        comparison = np.equal(y_truth, y_pred)
        accuracy = (np.where(comparison == True)[0].size)/comparison.size
        return accuracy