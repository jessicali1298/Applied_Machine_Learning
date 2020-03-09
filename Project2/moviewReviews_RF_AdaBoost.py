import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from random import shuffle

from nltk.stem import PorterStemmer
import nltk

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
all_ls = []

# read txt files and save as strings, put txt files in a string list
for i in range(len(test_path_ls)):

    with open(test_path_ls[i], 'r') as file:
        temp_test = file.read().replace('\n', '')

    with open(train_path_ls[i], 'r') as file:
        temp_train = file.read().replace('\n', '')
    
    test_ls.append([temp_test,i%2])
    train_ls.append([temp_train,i%2])
    
all_ls = train_ls + test_ls  

#%%
# Preprocess text info for training later
analyzer = TfidfVectorizer().build_analyzer()
stemmer= PorterStemmer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

# tfidf vectorizer with word stemmer
#tfidf_vect = TfidfVectorizer(analyzer=stemmed_words, stop_words='english')


# Initialize the vectorizers and classifiers
count_vect = CountVectorizer(analyzer='word', stop_words='english')
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_transformer = TfidfTransformer()

# Shuffle the data before preparing training and testing datasets
test_ls_shuffle = np.asarray(test_ls.copy())
shuffle(test_ls_shuffle)

train_ls_shuffle = np.asarray(train_ls.copy())
shuffle(train_ls_shuffle)

all_ls_shuffle = np.asarray(all_ls.copy())
shuffle(all_ls_shuffle)

train_data = train_ls_shuffle[:,0]
train_labels = train_ls_shuffle[:,1]

test_data = test_ls_shuffle[:,0]
test_labels = test_ls_shuffle[:,1]

all_data = all_ls_shuffle[:,0]
all_labels = all_ls_shuffle[:,1]

# for manual classification without pipeline (used for RandomSearchCV)
X_train_tfidf = tfidf_vect.fit_transform(train_data).toarray()

X_all = tfidf_vect.fit_transform(all_data).toarray()
print("all data counts: ", X_all.shape) 

# Split data into training and testing
X_train = X_all[0:len(train_ls)]
X_test = X_all[len(train_ls):len(all_ls)]



#%%
# Define base estimators for AdaBoost
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

svm = LinearSVC(random_state=0, tol=1e-5)
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
mnb = MultinomialNB()

#%%
RFC = RandomForestClassifier(n_estimators=300, bootstrap=False)
adaBoost = AdaBoostClassifier(base_estimator=svm, n_estimators=100, algorithm='SAMME')


#%%
## Manual checking
#adaBoost.fit(X_train, train_labels)
#predicted = adaBoost.predict(X_test)
#accuracy = adaBoost.score(X_test, test_labels)
#print(accuracy)
#print(np.mean(predicted == test_labels))

#%%
# cross validation using training/validation set
#cv_results = cross_validate(RFC, X_train_tfidf, train_labels, cv=5)
#print("cv results: ", cv_results['test_score'], '\n', 
#      "cv avg accuracy: ", np.mean(cv_results['test_score']))

from sklearn.pipeline import Pipeline
start_pip = time.time()
text_clf = Pipeline([('vect', tfidf_vect),
                     ('clf', RFC)])
text_clf.fit(train_data, train_labels)

predicted = text_clf.predict(test_data)
pip_duration = time.time() - start_pip
accuracy = text_clf.score(test_data, test_labels)
print(accuracy)
print(np.mean(predicted == test_labels))
print("computation time: ", pip_duration)









