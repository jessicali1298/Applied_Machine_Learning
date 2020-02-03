import numpy as np
import matplotlib as plt
import pandas as pd


# Logistic Regression:
# 1. Use gradient descent to find weights for log-odds ratio
# 2. Compute predicted probability (sigmoid function)
# 3 prob > 0.5, y = 1;   prob < 0.5, y = 0;   prob = 0.5, use random decider


#------------------PSEUDO-CODE-------------------
def gradient(X,y,w):
    N,D = X.shape
    yh = logistic(np.dot(X,w))
    grad = np.dot(X.T, th-y) / N
    return grad

def gradientDescent(X, y, w, a, end_cond):
    N,D = X.shape
    w = np.zeros(D)
    g = np.inf
    
    while np.linalg.norm(g) > end_cond:
        g = gradient(X,y,w)
        w = w - lr*g
    return w

def fit(X, y, a, end_cond):
    #CODE
    

def predict(X, a, end_cond):
    #CODE
    
def evaluate(y_truth, y_pred):
    #CODE