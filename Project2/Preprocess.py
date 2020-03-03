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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/'

#%% 
# Define a dummy tokenizer for TF-IDF Vectorizer
def dummy_tok(input):
    return input

#%%
from sklearn.datasets import fetch_20newsgroups
twenty_train_all = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
all_names = twenty_train_all.target_names
#all_labels = np.unique(twenty_train_all.target)

train_data = twenty_train_all.data
labels = twenty_train_all.target

#print(twenty_train_all.data[0])
#
print(all_names)
#print(all_labels)


# only test with a few groups
#categories = ['comp.graphics', 'sci.med']
#twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
#                                  shuffle=True, random_state=42)
#train_data = twenty_train.data
#labels = twenty_train.target




#%%
# Text-Preprocessing Tasks
# 1. converting all letters to lower or upper case
# 2. converting numbers into words or removing numbers
# 3. removing punctuations, accent marks and other diacritics
# 4. removing white spaces
# 5. expanding abbreviations
# 6. removing stop words, sparse terms, and particular words
# 7. text canonicalization (use only root words)

result = []
for data in train_data:
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
    
    # 6. text canonicalization (only use root words)
    stemmer= PorterStemmer()
    result_word = []

    for word in filtered_words:
        root_word = stemmer.stem(word)
        result_word.append(root_word)

    result.append(result_word)
    
    
vectorizer = TfidfVectorizer(analyzer='word', tokenizer = dummy_tok, 
                             preprocessor = dummy_tok,
                             token_pattern = None) 
X = vectorizer.fit_transform(result).toarray()
features = vectorizer.get_feature_names()

y = labels

# Initialize the Classifier and start training
RFC = RandomForestClassifier(n_estimators=300, random_state=0)
cv_results = cross_validate(RFC, X, y, cv=5)
print(cv_results['test_score'])






