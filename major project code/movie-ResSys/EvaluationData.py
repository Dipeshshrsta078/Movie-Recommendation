# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:46:13 2019

@author: Heriz
"""

from surprise.model_selection import train_test_split

class EvaluationData:
    
    def __init__(self, data, popularityRankings):
        
        self.rankings = popularityRankings
        
        #Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        #Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)
            
    def GetFullTrainSet(self):
        return self.fullTrainSet
    
#    def GetFullAntiTes tSet(self):
#        return self.fullAntiTestSet
    
    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet
    
    def GetPopularityRankings(self):
        return self.rankings