import numpy as np
import pandas as pd
class Data_Cleaner:
    
    def data_report(self, input_data, dataset_name):
        print('\n----DATA REPORT OF ',dataset_name, '----')
        
        # 1. check data types of all features of input dataset
        data_types = input_data.dtypes
        print('\nfeature dtypes: ', '\n', data_types)
        
        # 2. check missing values
        missing = np.any(pd.isnull(input_data),axis=1)
        missing_idx = np.unique(np.where(missing == True)[0])
        print('\ninstances with MISSING values: ', '\n', missing_idx)
        
        # 3. check if there are duplicates
        duplicates = np.unique(np.where(input_data.duplicated() == True)[0])
        print('\nDUPLICATED instances: ', '\n', duplicates)
        
        # 4. check malformed values (mainly in dataset2,3)
        malformed = np.unique(np.where((input_data == '?'))[0])
        print('\nMALFORMED instances: ', '\n', malformed)
        
        
    
        
        
    #%%
    def data_stats(self, input_data):
        # 5. number of pos. and neg. classes
        counts = input_data.iloc[:,input_data.shape[1]-1].value_counts()
        print('\n# of binary classifications: ', '\n', counts)
              
        # 6. Distribution of numerical features (min, max, mean, range)
        data_types = input_data.dtypes
        numeric_feature_idx = np.where((data_types == int) | (data_types == float))[0]
        
        numeric_max = []
        numeric_min = []
        numeric_mean = []
        numeric_range = []
        
        
        for i in range(numeric_feature_idx.shape[0]):
            numeric = input_data.iloc[:,numeric_feature_idx[i]].to_numpy()
#            print(numeric)
            num_max = np.max(numeric)
            num_min = np.min(numeric)
            num_mean = np.mean(numeric)
            num_range = num_max - num_min
            
            numeric_max.append(num_max)
            numeric_min.append(num_min)
            numeric_mean.append(num_mean)
            numeric_range.append(num_range)
            
        # numeric_analysis = np.c_(numeric_max, numeric_min, numeric_mean, numeric_range)
        analysis_df = pd.DataFrame({'max':numeric_max, 'min': numeric_min,
                                    'mean':numeric_mean, 'range':numeric_range})
        
        
        print('\n', analysis_df)
        return analysis_df
    
    def normalize(self, input_col):
        
        input_mean = np.mean(input_col)
        input_range = np.max(input_col) - np.min(input_col)
        output_col = (input_col - input_mean)/input_range
        return output_col
        
        
        
        
        
    def clean_data(self, input_data): 
        # 1. find bad data
        missing = np.any(pd.isnull(input_data),axis=1)
        missing_idx = np.unique(np.where(missing == True)[0])
        
        duplicates = np.unique(np.where(input_data.duplicated() == True)[0])
        
        malformed = np.unique(np.where(input_data == '?')[0])
        
        # 2. remove bad data
        bad_data_idx = np.concatenate((missing_idx, duplicates, malformed), axis = None)
        
        # find empty indices in bad_data_idx
        empty_idx = []
        for i in range(bad_data_idx.size):
            if bad_data_idx[i] is None:
                empty_idx.append(i)
        
        # removes all empty indices from bad_data_idx
        bad_data_idx = np.delete(bad_data_idx,empty_idx)
        
        input_data = input_data.drop(bad_data_idx)
        
        output_data = input_data.reset_index(drop=True)
        
        
        return output_data
        
    
    #%%
    # change categorical variable to numerical
    def cat_to_num(self, input_data):
        output_data = input_data.copy()
        
        data_types = input_data.dtypes
        
        # find where non-numeric variables are
        category_var = np.where((data_types != int) & (data_types != float) & (data_types != bool))[0]
        
        # assign unique number code to categorical variables
        for var in category_var:
            codes, uniques = pd.factorize(input_data.iloc[:,var])
            
            # replace original categorical vars with numeric values
            output_data.iloc[:,var] = codes
        
        # print indices of categorical variables/features
        print('\nIndices of Categorical Variables: ', category_var)
        return output_data
    
    def add_one(self, input_data):
        input_data = input_data.insert(0, 'a', np.ones(len(input_data.index)))
     
    def extract_y(self, input_data):
        y = input_data.iloc[:,len(input_data.columns)-1]
        return y
    
        
    #%%
    def data_prep(self, input_data, dataset_name):
        # add a column of 1 to train data to get bias term for log reg
        self.add_one(input_data)
        self.data_report(input_data, dataset_name)
        cleaned_data = self.clean_data(input_data)
        final_data = self.cat_to_num(cleaned_data)
        print(final_data.iloc[0:9,:])
        return final_data

    def data_prep_naive(self, input_data):
        output_data = input_data.drop(input_data.columns[0], axis=1).to_numpy()
        return output_data
    
        
        
        
        
        