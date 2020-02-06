import pandas as pd
import numpy as np
import Data_Cleaner as cd
import Log_Regression as lgr
from sklearn.linear_model import LogisticRegression

class Cross_Validation: 
    
    def dataset_speration(dataset, k):
        