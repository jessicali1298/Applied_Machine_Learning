import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import re
import string

# import and download NLP Resources
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#%%
# Import 20news dataset
from sklearn.datasets import fetch_20newsgroups

#-----------------------test all 20 groups--------------------------------
twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                  shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                 shuffle=True, random_state=42)
twenty_all = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                                 shuffle=True, random_state=42)

#-----------------------only test with a few groups------------------------
#categories = ['alt.atheism','comp.graphics', 'misc.forsale','rec.autos', 'sci.med',
#              'soc.religion.christian','talk.politics.guns','talk.politics.mideast']
#categories = ['alt.atheism','comp.graphics', 'rec.autos', 'sci.med']
#categories = ['comp.graphics', 'sci.med']

#twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
#                                  shuffle=True, random_state=42)
#twenty_test = fetch_20newsgroups(subset='test', categories=categories, 
#                                 shuffle=True, random_state=42)
#twenty_all = fetch_20newsgroups(subset='all', categories=categories, 
#                                 shuffle=True, random_state=42)

train_data = twenty_train.data
train_labels = twenty_train.target

test_data = twenty_test.data
test_labels = twenty_test.target

all_data = twenty_all.data
all_labels = twenty_all.target



#%%
# Preprocess text info for training later
analyzer = TfidfVectorizer().build_analyzer()
stemmer= PorterStemmer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

#tfidf_vect = TfidfVectorizer(analyzer=stemmed_words, stop_words='english')


# Initialize the vectorizers and classifiers
count_vect = CountVectorizer(analyzer='word', stop_words='english')
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 100000)
tfidf_transformer = TfidfTransformer()

# for our own info, check how many documents and different words there are
# in the dataset used
#X_train_counts = count_vect.fit_transform(twenty_train.data)
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf = tfidf_vect.fit_transform(train_data)
print("train data counts: ", X_train_tfidf.shape)


# for manual classification without pipeline (used for RandomSearchCV)

X_all = tfidf_vect.fit_transform(all_data).toarray()
print("all data counts: ", X_all.shape) 

# Split data into training and testing
X_train = X_all[0:len(train_data)]
X_test = X_all[len(train_data):len(X_all)]

# Initialize the Classifier and start training

#cv_results = cross_validate(clf, train_data, train_labels, cv=5)
#print(cv_results['test_score'])


#%%
# Initialize the Classifier and start training
svm_final = LinearSVC(tol = 1e-5, random_state = 0, multi_class = 'ovr', 
                      max_iter = 4000, class_weight = 'balanced')
mnb_final = MultinomialNB(alpha = 0.01)
clf = LinearSVC()
MNB = MultinomialNB()
#%%
## Define parameters used for parameter opimization
##
###--------------------------------Linear SVM----------------------------------
###tolerance for stopping criteria
#tol = [1e-5, 2e-5, 5e-5, 1e-4]
#multi_class = ['ovr', 'crammer_singer']
##maximum iterations
#max_iter = [int(x) for x in np.linspace(start = 1000, stop = 5000, num = 5)]
#
## Create the random grid
#grid_svm = {'tol': tol,
#            'multi_class': multi_class,
#            'max_iter': max_iter
#            }
#
#
##--------------------------Multinomial Naive Bayes----------------------------
#
## Smoothing parameter
#alpha = [0.01, 0.1, 1, 2, 5]
#
#
## Create the random grid
#grid_mnb = {'alpha': alpha
#            }
##%%
## Perform random parameter search
#from sklearn.model_selection import RandomizedSearchCV
#
##--------------------------------Linear SVM------------------------------------
#start_ran_svm = time.time()
#svm_random = RandomizedSearchCV(estimator = clf, param_distributions = grid_svm, 
#                               n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
## Fit the random search model
#svm_random.fit(X_train, train_labels)
#
#svm_random.predict(X_test)
#ran_svm_duration = time.time() - start_ran_svm
#accuracy_svm = svm_random.score(X_test,test_labels)
#print(svm_random.best_params_)
#print(svm_random.best_score_)
#print(accuracy_svm)
#print("RandomSearch SVM duration: ", ran_svm_duration)
#
##--------------------------Multinomial Naive Bayes-----------------------------
#start_ran_mnb = time.time()
#mnb_random = RandomizedSearchCV(estimator = MNB, param_distributions = grid_mnb, 
#                               n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#
## Fit the random search model
#mnb_random.fit(X_train, train_labels)
#mnb_random.predict(X_test)
#ran_mnb_duration = time.time() - start_ran_mnb
#accuracy_mnb = mnb_random.score(X_test,test_labels)
#
#print(mnb_random.best_params_)
#print(mnb_random.best_score_)
#print(accuracy_mnb)
#print("RandomSearch Ada duration: ", ran_mnb_duration)

#%%
## Perform Bayesian Optimization for parameters
#from skopt import BayesSearchCV
#
##--------------------------------Linear SVM------------------------------------
#start_bayes_svm = time.time()
#bayes_svm = BayesSearchCV(clf, grid_svm, n_iter=50, cv=3)
#
#bayes_svm.fit(X_train, train_labels)
#bayes_svm.predict(X_test)
#bayes_svm_duration = time.time() - start_bayes_svm
#accuracy_bayes = bayes_svm.score(X_test, test_labels)
#print(bayes_svm.best_params_)
#print(bayes_svm.best_score_)
#print(accuracy_bayes)
#print("Bayes Op SVM duration: ",bayes_svm_duration)
#
##--------------------------Multinomial Naive Bayes-----------------------------
#start_bayes_mnb = time.time()
#bayes_mnb = BayesSearchCV(MNB, grid_mnb, n_iter=10, cv=3)
#bayes_mnb.fit(X_train, train_labels)
#bayes_mnb_duration = time.time() - start_bayes_mnb
#print(bayes_mnb.best_params_)
#print(bayes_mnb.best_score_)
#print("Bayes Op Ada duration: ", bayes_mnb_duration)

#%%
# cross validation using training/validation set

from sklearn.pipeline import Pipeline
cv_results = cross_validate(svm_final, X_train_tfidf, train_labels, cv=5)
print("SVM results: ", cv_results['test_score'], '\n', 
      "SVM avg accuracy: ", np.mean(cv_results['test_score']))
start_pip = time.time()
text_clf = Pipeline([('vect', tfidf_vect),
                     ('clf', svm_final)])
text_clf.fit(train_data, train_labels)

predicted = text_clf.predict(test_data)
pip_duration = time.time() - start_pip

accuracy = text_clf.score(test_data, test_labels)
print(accuracy)
print("SVM testing accuracy: ", np.mean(predicted == test_labels))
print("computation time SVM: ", pip_duration)


cv_results = cross_validate(mnb_final, X_train_tfidf, train_labels, cv=5)
print("MNB results: ", cv_results['test_score'], '\n', 
      "MNB avg accuracy: ", np.mean(cv_results['test_score']))

start_pip2 = time.time()
text_clf = Pipeline([('vect', tfidf_vect),
                     ('clf', mnb_final)])
text_clf.fit(train_data, train_labels)

predicted2 = text_clf.predict(test_data)
pip_duration2 = time.time() - start_pip2

#accuracy = text_clf.score(test_data, test_labels)
#print(accuracy)
print("MNB testing accuracy: ", np.mean(predicted2 == test_labels))
print("computation time MNB: ", pip_duration2)


# calculate confustion matrix
conf = confusion_matrix(test_labels, predicted)
plt.figure()
plt.imshow(conf)
plt.title("Confusion Matrix - 20NewsGroup SVM"), plt.xticks([]), plt.yticks([])
plt.show()

conf2 = confusion_matrix(test_labels, predicted2)
plt.figure()
plt.imshow(conf2)
plt.title("Confusion Matrix - 20NewsGroup Multinomial NB"), plt.xticks([]), plt.yticks([])
plt.show()