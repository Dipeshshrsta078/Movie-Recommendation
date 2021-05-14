# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:42:10 2019

@author: Heriz
"""
from MovieLens import MovieLens
from collections import defaultdict
from six import iteritems
import numpy as np

class Trainset:
    def __init__(self):
        ml = MovieLens()
        self.dataset = ml.getTrainSet()
        userVal = int(self.dataset['userId'])
        self.n_users = len(userVal)
        self.n_items = len(set(self.dataset['movieId']))
        
        self._global_mean = None
    
        self.ur = defaultdict(list)
        for index,row in self.dataset.iterrows():
            userId = row['userId']
            movieId = row['movieId']
            ratings = row['rating']
            self.ur[userId].append((movieId,ratings))
            
        self.ir = defaultdict(list)
        for index,row in self.dataset.iterrows():
            userId = row['userId']
            movieId = row['movieId']
            ratings = row['rating']
            self.ur[movieId].append((userId,ratings))
    
    
    def knows_user(self,uid):
        if (uid in self.ur):
            return True 
        else:
            return False
        
    def knows_item(self,iid):
        if (iid in self.ir):
            return True 
        else:
            return False
    
    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids (see
            :ref:`this note <raw_inner_note>`).
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r
    
    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)
    
    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean