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

from skopt import BayesSearchCV


#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/'

#%% 
# Define a dummy tokenizer for TF-IDF Vectorizer
def dummy_tok(input):
    return input

#%%
from sklearn.datasets import fetch_20newsgroups

#-----------------------only test with a few groups------------------------
categories = ['comp.graphics', 'sci.med']
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
result = []
for data in all_data:
    # 1. convert all letters to lower cases
    temp = data.lower()
    
    # 2. remove numbers
    temp = re.sub(r'\d+', '', temp)

    # 3. remove punctuations, accent marks and other diacritics
    temp = temp.translate(str.maketrans('','',string.punctuation))

    # 4. remove leading and ending white spaces
    temp = temp.strip()
    
    # 5. remove stop words
    stops = set(stopwords.words("english"))
    tokens = word_tokenize(temp)
    filtered_words = [word for word in tokens if word not in stops]
    
    # 6. word stemming (only use root words)
    stemmer= PorterStemmer()
    result_word = []

    for word in filtered_words:
        root_word = stemmer.stem(word)
        result_word.append(root_word)

    result.append(result_word)
    
    
#%%
#result_test = []
#for data in test_data:
#    # 1. convert all letters to lower cases
#    temp = data.lower()
#    
#    # 2. remove numbers
#    temp = re.sub(r'\d+', '', temp)
#
#    # 3. remove punctuations, accent marks and other diacritics
#    temp = temp.translate(str.maketrans('','',string.punctuation))
#
#    # 4. remove leading and ending white spaces
#    temp = temp.strip()
#    
#    # 5. remove stop words
#    stops = set(stopwords.words("english"))
#    tokens = word_tokenize(temp)
#    filtered_words = [word for word in tokens if word not in stops]
#    
#    # 6. word stemming (only use root words)
#    stemmer= PorterStemmer()
#    result_word = []
#
#    for word in filtered_words:
#        root_word = stemmer.stem(word)
#        result_word.append(root_word)
#
#    result_test.append(result_word)
    
#%%    
vectorizer = TfidfVectorizer(analyzer='word', tokenizer = dummy_tok, 
                             preprocessor = dummy_tok,
                             token_pattern = None)

X_all = vectorizer.fit_transform(result).toarray()
#print(X_all.shape) #1963

# Split data into training and testing
X_train = X_all[0:len(train_data)]
X_test = X_all[len(train_data):len(X_all)]

# Initialize the Classifier and start training
RFC = RandomForestClassifier(n_estimators=300, random_state=0)
#cv_results = cross_validate(RFC, X, y, cv=5)

RFC.fit(X_train, train_labels)
predicted = RFC.predict(X_test)

accuracy = RFC.score(X_test, test_labels)
print(accuracy)  #0.48025477

#print(X.shape) #(1178, 18399)
#print(len(result)) #(1178)
#print(cv_results['test_score']) #(0.9449, 0.9534, 0.9194, 0.9194, 0.9529)



