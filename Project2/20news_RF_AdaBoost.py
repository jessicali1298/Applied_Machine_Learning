import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# import and download NLP Resources
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



#%%
# Import 20news dataset
from sklearn.datasets import fetch_20newsgroups

#-----------------------test all 20 groups--------------------------------
#twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
#                                  shuffle=True, random_state=42)
#twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
#                                 shuffle=True, random_state=42)
#twenty_all = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
#                                 shuffle=True, random_state=42)

#-----------------------only test with a few groups------------------------
categories = ['alt.atheism','comp.graphics', 'misc.forsale','rec.autos', 'sci.med',
              'soc.religion.christian','talk.politics.guns','talk.politics.mideast']
#categories = ['alt.atheism','comp.graphics', 'rec.autos', 'sci.med']
#categories = ['comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                  categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                 categories=categories, shuffle=True, random_state=42)
twenty_all = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                                categories=categories, shuffle=True, random_state=42)

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

# tfidf vectorizer with word stemmer
tfidf_vect = TfidfVectorizer(analyzer=stemmed_words, stop_words='english')


# Initialize the vectorizers and classifiers
count_vect = CountVectorizer(analyzer='word', stop_words='english')
#tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_transformer = TfidfTransformer()

# for our own info, check how many documents and different words there are
# in the dataset used
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_counts.shape)
print(X_train_tfidf.shape)
print(len(train_data))


# for manual classification without pipeline (used for RandomSearchCV)

X_all = tfidf_vect.fit_transform(all_data).toarray()
print(X_all.shape) 

# Split data into training and testing
X_train = X_all[0:len(train_data)]
X_test = X_all[len(train_data):len(X_all)]

# Initialize the Classifier and start training
RFC_final = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf = 1,
                             max_features='auto', max_depth = None, bootstrap=False, random_state=0)
RFC = RandomForestClassifier(random_state=0)
adaBoost = AdaBoostClassifier(random_state=0)






#cv_results = cross_validate(RFC, X, y, cv=5)
#print(cv_results['test_score'])


#%%
# Define parameters used for parameter opimization

#--------------------------------RANDOM FORESET--------------------------------
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 8)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
grid_rfc = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
            }


#----------------------------------ADABOOST------------------------------------
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

svm = LinearSVC(random_state=0, tol=1e-5)
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
mnb = MultinomialNB()

# Define base estimators for AdaBoost
base_estimator = [svm, lr, mnb, None]

# Number of features to consider at every split
algorithm = ['SAMME', 'SAMME.R']

# Create the random grid
grid_ada = {'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'algorithm': algorithm
            }

#%%
# Perform random parameter search
from sklearn.model_selection import RandomizedSearchCV

#--------------------------------RANDOM FORESET--------------------------------
#rf_random = RandomizedSearchCV(estimator = RFC, param_distributions = grid_rfc, 
#                               n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
## Fit the random search model
#rf_random.fit(X_train, train_labels)
#
#rf_random.predict(X_test)
#accuracy_rfc = rf_random.score(X_test,test_labels)
#print(rf_random.best_params_)
#print(rf_random.best_score_)
#print(accuracy_rfc)

#------------------------------------ADABOOST----------------------------------
#
#rf_random = RandomizedSearchCV(estimator = adaBoost, param_distributions = grid_ada, 
#                               n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#
## Fit the random search model
#rf_random.fit(X_train, train_labels)
#rf_random.predict(X_test)
#accuracy_ada = rf_random.score(X_test,test_labels)
#
#print(rf_random.best_params_)
#print(rf_random.best_score_)
#print(accuracy_ada)


#%%
# Perform Bayesian Optimization for parameters
from skopt import BayesSearchCV

#--------------------------------RANDOM FORESET--------------------------------
#bayes_opt = BayesSearchCV(RFC, grid_rfc, n_iter=32, cv=3)
#
#bayes_opt.fit(X_train, train_labels)
#print(bayes_opt.best_params_)
#print(bayes_opt.best_score_)

#------------------------------------ADABOOST----------------------------------
bayes_opt = BayesSearchCV(adaBoost, grid_ada, n_iter=10, cv=3)

bayes_opt.fit(X_train, train_labels)
print(bayes_opt.best_params_)
print(bayes_opt.best_score_)


#%%
#from sklearn.pipeline import Pipeline
#text_clf = Pipeline([('vect', tfidf_vect),
##                     ('tfidf', tfidf_transformer),
#                     ('clf', RFC_final)])
#text_clf.fit(train_data, train_labels)
#
#predicted = text_clf.predict(test_data)
##accuracy = text_clf.score(test_data, test_labels)
##print(accuracy)
#print(np.mean(predicted == test_labels))

