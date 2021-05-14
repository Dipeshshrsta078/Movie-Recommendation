# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:51:00 2019

@author: Heriz
"""

from MovieLens import MovieLens
#from .algo_base import AlgoBase
from SVD import SVD
from Evaluator import Evaluator
import pickle
import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
#    print("\nComputing movie popularity ranks")
    rankings = ml.getPopularityRanks()

#    popular = []
#    for item,value in rankings.items():
#        popular.append((item,ml.getMovieName(item),value))
#    pickle_out = open('popularMovie.pickle','wb')
#    pickle.dump(popular,pickle_out)
#    pickle_out.close()
#    
#    print(popular[:10])
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml,k=55,verbos = True)
    
    

