import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%%
# import the movie review txt files
test_path_neg = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/aclImdb/test/neg/'
test_path_pos = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/aclImdb/test/pos/'
train_path_neg = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/aclImdb/train/neg/'
train_path_pos = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/aclImdb/train/pos'

test_dir_n = os.listdir(test_path_neg)
test_dir_p = os.listdir(test_path_pos)
train_dir_n = os.listdir(train_path_neg)
train_dir_p = os.listdir(train_path_pos)

test_path_ls = []
train_path_ls = []
for i in range(len(test_dir_n)):
    subpath_test_n = os.path.join(test_path_neg,test_dir_n[i])
    subpath_test_p = os.path.join(test_path_pos,test_dir_p[i])
    subpath_train_n = os.path.join(train_path_neg,train_dir_n[i])
    subpath_train_p = os.path.join(train_path_pos,train_dir_p[i])
    
    test_path_ls.append(subpath_test_n)
    test_path_ls.append(subpath_test_p)
    train_path_ls.append(subpath_train_n)
    train_path_ls.append(subpath_train_p)

test_ls = []
train_ls = []

# read txt files and save as strings, put txt files in a string list
for i in range(len(test_path_ls)):
    with open(test_path_ls[i], 'r') as file:
        temp_test = file.read().replace('\n', '')

    with open(train_path_ls[i], 'r') as file:
        temp_train = file.read().replace('\n', '')
        
    test_ls.append(temp_test)
    train_ls.append(temp_train)

    








