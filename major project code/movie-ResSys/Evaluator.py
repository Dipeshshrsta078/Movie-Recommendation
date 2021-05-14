# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:50:08 2019

@author: Heriz
"""
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
import pickle
#import random

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
    
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
    
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            if(doTopN):
                print("Evaluating ", algorithm.GetName(), "...")
            else:
                print("\nUsing recommender SVD")
                print("\nBuilding recommendation model...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        if(doTopN):
            print("\n")
            print("\nLegend:\n")
            print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
            print("{:<10} {:<10}".format("Algorithm", "RMSE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f}".format(name, metrics["RMSE"]))
        
        
    def SampleTopNRecs(self, ml, testSubject=5, k=10,verbos = True):
        
        for algo in self.algorithms:
            if(verbos):
                print("\nUsing recommender SVD")
                print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            # algo.GetAlgorithm().fit(trainSet)
            print("Analysis Complete...")
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithm().test(testSet)
            
            recommendations = []
            
            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                if estimatedRating >= 4.0:
                    intMovieID = int(movieID)
                    recommendations.append((intMovieID, estimatedRating))
            
            pickle_out = open(str(testSubject)+".pickle","wb")
            pickle.dump(recommendations,pickle_out)
            pickle_out.close()
            
            pickle_in = open(str(testSubject)+".pickle","rb")
            recommendations = pickle.load(pickle_in)
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])