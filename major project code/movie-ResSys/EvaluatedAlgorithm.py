# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:49:32 2019

@author: Heriz
"""
from RecommenderMetrics import RecommenderMetrics

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=False):
        '''
            this function calculates RMSE score
        '''
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet()) #train our model
        predictions = self.algorithm.test(evaluationData.GetTestSet()) #get predicted output
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions) #get RMSE score
        
        if (verbose):
            print("Analysis complete.")
    
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
    
    
