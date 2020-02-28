import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import re
import string

root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/'

#%%
from sklearn.datasets import fetch_20newsgroups
#twenty_train_all = fetch_20newsgroups(subset='train')
#all_names = twenty_train_all.target_names
#all_labels = np.unique(twenty_train_all.target)
#
#print(twenty_train_all.data[0])
#
#print(all_names)
#print(all_labels)


# only test with a few groups
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
                                  shuffle=True, random_state=42)
train_data = twenty_train.data
labels = twenty_train.target




#%%
# Text-Preprocessing Tasks
# 1. converting all letters to lower or upper case
# 2. converting numbers into words or removing numbers
# 3. removing punctuations, accent marks and other diacritics
# 4. removing white spaces
# 5. expanding abbreviations
# 6. removing stop words, sparse terms, and particular words
# 7. text canonicalization (use only root words)

clean_train_data = []

for data in train_data:
    # 1. convert all letters to lower
    temp = data.lower()
    
    # 2. convert numbers into words or removing numbers
    temp = re.sub(r’\d+’, ‘’, temp)

    # 3. removing punctuations, accent marks and other diacritics
    temp = temp.translate(string.maketrans(“”,””), string.punctuation)

    # 4. removing white spaces
    temp = temp.strip()
    
    # 5. expanding abbreviations
    











