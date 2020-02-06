import numpy as np
from math import exp,log,pi,sqrt

class Naive_Bayes:
    
    def getNeededValuesPerClass(self, dataset):
    	summary = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)] #get all the rows together no matter how many columns
    	del(summary[-1]) #the lable is no longer needed
    	return summary
    
    def breakDown(self, dataset):
    	rowsPerClass = dict()
    	for i in range(len(dataset)):
    		eachRow = dataset[i]
    		lable = eachRow[-1] #the last entry is always the lable
    		if (lable not in rowsPerClass):
    			rowsPerClass[lable] = list() #create the list for each category to hold the values
    		rowsPerClass[lable].append(eachRow)
    	VariablesPerFeature = dict() #hold the mean and std per feature where each key is the class
    	for classValue, rows in rowsPerClass.items():
    		VariablesPerFeature[classValue] = self.getNeededValuesPerClass(rows) #do all the calcuation per category
    	return VariablesPerFeature
    
    
    #gaussian assumption
    def calculateProbability(self, x, mean, stdev):
        exponent = np.exp(-(np.power((x-mean),2) / (2 * np.power(stdev,2))))
        result = (1/(np.sqrt(2*np.pi) * stdev)) * exponent
        return result
    
    # Calculate the probabilities of predicting each class for a given row
    def calculateClassProbabilities(self, summaries, row, totalRowsInDataset):
    	print("total number of rows in the data set", totalRowsInDataset) 
    	probabilities = dict()
    	for category, classSummaries in summaries.items():
    		probabilities[category] = log(summaries[category][0][-1]/float(totalRowsInDataset)) # this adds the prior (always the first row cause they should be the same for all)
    		for i in range(len(classSummaries)):
    			mean, stdev, count = classSummaries[i]
    			probabilities[category] += log(self.calculateProbability(row[i], mean, stdev)) # collection per feature per class
    	return probabilities
    
    
    def fit(self, dataset):
        varPerFeature = self.breakDown(dataset)
        return varPerFeature
    

    def predict(self, summaries, row, totalRowsInDataset):
        prob = self.calculateClassProbabilities(summaries, row, totalRowsInDataset)
        return prob
    

