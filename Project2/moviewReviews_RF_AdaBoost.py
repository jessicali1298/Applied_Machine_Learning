import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import re
import string

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
root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project2/acllmdb'