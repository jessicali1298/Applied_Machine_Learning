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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#%%
from sklearn.datasets import fetch_20newsgroups
#twenty_train_all = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
#all_names = twenty_train_all.target_names
#
#train_data = twenty_train_all.data
#labels = twenty_train_all.target
#
#print(all_names)


#-----------------------only test with a few groups------------------------
categories = ['alt.atheism','comp.graphics', 'misc.forsale','rec.autos', 'sci.med',
              'soc.religion.christian','talk.politics.guns','talk.politics.mideast']
#categories = ['comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
                                  shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, 
                                 shuffle=True, random_state=42)
twenty_all = fetch_20newsgroups(subset='all', categories=categories, 
                                 shuffle=True, random_state=42)
train_data = twenty_train.data
train_labels = twenty_train.target

test_data = twenty_test.data
test_labels = twenty_test.target

all_data = twenty_all.data
all_labels = twenty_all.target



#%%
count_vect = CountVectorizer(analyzer='word', stop_words='english')

tfidf_transformer = TfidfTransformer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_counts.shape)
print(X_train_tfidf.shape)
print(len(train_data))
#print(count_vect.get_feature_names())



# Initialize the Classifier and start training
RFC = RandomForestClassifier(n_estimators=300, random_state=0)
#cv_results = cross_validate(RFC, X, y, cv=5)
#print(cv_results['test_score'])

#%%
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', count_vect),
                     ('tfidf', tfidf_transformer),
                     ('clf', RFC)])
text_clf.fit(train_data, train_labels)

predicted = text_clf.predict(test_data)
print(np.mean(predicted == test_labels))

