# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:49:05 2019

@author: Heriz
"""
#from collections import defaultdict
import numpy as np

class RecommenderMetrics:

    def RMSE(predictions):
        if not predictions:
#            raise Va lueError('Prediction list is empty.')
            print("Prediction list is empty")

        mse = np.mean([float((true_r - est)**2)
                       for (_, _, true_r, est, _) in predictions])
        rmse_ = np.sqrt(mse)
        return rmse_

#    def GetTopN(predictions, n=10, minimumRating=4.0):
#        topN = defaultdict(list)
#        for userID, movieID, actualRating, estimatedRating, _ in predictions:
#            if (estimatedRating >= minimumRating):
#                topN[int(userID)].append((int(movieID), estimatedRating))
#
#        for userID, ratings in topN.items():
#            ratings.sort(key=lambda x: x[1], reverse=True)
#            topN[int(userID)] = ratings[:n]
#
#        return topN