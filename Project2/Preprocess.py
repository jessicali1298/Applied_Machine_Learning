import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/'

#%%
from sklearn.datasets import fetch_20newsgroups
twenty_train_all = fetch_20newsgroups(subset='train')
all_names = twenty_train_all.target_names
all_labels = np.unique(twenty_train_all.target)

print(twenty_train_all.data[0])

print(all_names)
print(all_labels)


# only test with a few groups
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
                                  shuffle=True, random_state=42)




