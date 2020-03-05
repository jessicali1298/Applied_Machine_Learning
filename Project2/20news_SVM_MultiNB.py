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
twenty_train = fetch_20newsgroups(subset='train', 
                                  shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
                                 shuffle=True, random_state=42)
twenty_all = fetch_20newsgroups(subset='all',
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

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')


# Initialize the vectorizers and classifiers
count_vect = CountVectorizer(analyzer='word', stop_words='english')
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_transformer = TfidfTransformer()

# for our own info, check how many documents and different words there are
# in the dataset used
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_counts.shape)
print(X_train_tfidf.shape)
print(len(train_data))


# for manual classification without pipeline

X_all = tfidf_vect.fit_transform(all_data).toarray()
print(X_all.shape) #1963

# Split data into training and testing
X_train = X_all[0:len(train_data)]
X_test = X_all[len(train_data):len(X_all)]

# Initialize the Classifier and start training
clf = LinearSVC(random_state=0, tol=1e-5)
MNB = MultinomialNB()
#cv_results = cross_validate(clf, train_data, train_labels, cv=5)
#print(cv_results['test_score'])

#%%

# Perform grid search to find best parameters
#from sklearn.model_selection import RandomizedSearchCV
#
## Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 8)]
#
## Number of features to consider at every split
#max_features = ['auto', 'sqrt']
#
## Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
#
## Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
#
## Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
#
## Method of selecting samples for training each tree
#bootstrap = [True, False]
#
## Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}
#
#rf_random = RandomizedSearchCV(estimator = RFC, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
## Fit the random search model
#rf_random.fit(X_train, train_labels)
#
#print(rf_random.best_params_)


#%%
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', tfidf_vect),
#                     ('tfidf', tfidf_transformer),
                     ('clf', clf)])
text_clf.fit(train_data, train_labels)

predicted = text_clf.predict(test_data)
#accuracy = text_clf.score(test_data, test_labels)
#print(accuracy)
print("SVM: ", np.mean(predicted == test_labels))

text_clf_MNB = Pipeline([('vect', tfidf_vect),
#                     ('tfidf', tfidf_transformer),
                     ('clf', MNB)])
text_clf_MNB.fit(train_data, train_labels)

predicted = text_clf_MNB.predict(test_data)
#accuracy = text_clf.score(test_data, test_labels)
#print(accuracy)
print("Multinomial Naive Bayes: ", np.mean(predicted == test_labels))
