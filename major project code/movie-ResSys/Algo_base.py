# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:52:13 2019

@author: Heriz
"""

"""
The :mod:`surprise.prediction_algorithms.algo_base` module defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals )
import warnings

from six import get_unbound_function as guf
from Prediction import Prediction
from Prediction import PredictionImpossible

class AlgoBase(object):
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self):

        if (guf(self.__class__.fit) is guf(AlgoBase.fit) and
           guf(self.__class__.train) is not guf(AlgoBase.train)):
            warnings.warn('It looks like this algorithm (' +
                          str(self.__class__) +
                          ') implements train() '
                          'instead of fit(): train() is deprecated, '
                          'please use fit() instead.', UserWarning)

    def train(self, trainset):
        warnings.warn('train() is deprecated. Use fit() instead', UserWarning)

        self.skip_train = True
        self.fit(trainset)

        return self

    def fit(self, trainset):
        if (guf(self.__class__.train) is not guf(AlgoBase.train) and
                not self.skip_train):
            self.train(trainset)
            return
        self.skip_train = False

        self.trainset = trainset

        return self
    
    def test(self,testset,verbose = False):
        predictions = [self.predict(uid,iid,r,verbose = verbose)
                     for (uid,iid,r) in testset]
        return predictions
    
    def default_prediction(self):
        '''Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        '''

        return self.trainset.global_mean
    
    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
            
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)
        
        details = {}
        try:
            est = self.estimate(iuid, iiid)
        
            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est
        
            details['was_impossible'] = False
        
        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)
        
        # clip estimate into [lower_bound, higher_bound]
        if clip:
           lower_bound, higher_bound = self.trainset.rating_scale
           est = min(higher_bound, est)
           est = max(lower_bound, est)
        
        pred = Prediction(uid, iid, r_ui, est, details)
        
        if verbose:
            print(pred)
        
        return pred
    
    
    
    
    
   


