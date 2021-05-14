# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:07:21 2019

@author: Heriz
"""

import numpy as np
from Algo_base import AlgoBase
from Prediction import PredictionImpossible
from Prediction import Prediction
import pickle

class SVD(AlgoBase):

    def __init__(self):
        AlgoBase.__init__(self) 
    
    def fit(self,trainset):
        #look this
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)
    
    def sgd(self,trainset):
        #look this
        n_factors = 100  # number of factors to define user and item property
        alpha = .01  # learning rate
        n_epochs = 20  # number of iteration of the SGD procedure
        reg = 0.02 # for reguralization
        self.global_mean = float(self.trainset.global_mean)
        
        # Randomly initialize the user and item factors.
        # np.ndarray[n_users,n_factor]
        
        #user biases
        bu = np.zeros(trainset.n_users, np.double)
        #item biases
        bi = np.zeros(trainset.n_items, np.double)
        
        #user factors
        pu = np.random.normal(0, 0.1,
                        (trainset.n_users, n_factors))
        #item factors
        qi = np.random.normal(0, 0.1,
                        (trainset.n_items, n_factors))

        for current_epoch in range(n_epochs):
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = np.dot(qi[i],pu[u])
                err = r - (self.global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += alpha * (err - reg * bu[u])
                bi[i] += alpha * (err - reg * bi[i])

                # update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += alpha * (err * qif - reg * puf)
                    qi[i, f] += alpha * (err * puf - reg * qif)

        self.bu = bu
        pickle_outbu = open("bu.pickle","wb")
        pickle.dump(self.bu,pickle_outbu)
        pickle_outbu.close()
        
        self.bi = bi
        pickle_outbi = open("bi.pickle","wb")
        pickle.dump(self.bi,pickle_outbi)
        pickle_outbi.close()
        
        self.pu = pu
        pickle_outpu = open("pu.pickle","wb")
        pickle.dump(self.pu,pickle_outpu)
        pickle_outpu.close()
        
        self.qi = qi
        pickle_outqi = open("qi.pickle","wb")
        pickle.dump(self.qi,pickle_outqi)
        pickle_outqi.close()
    
    def estimate(self, u, i):
        # Should we cythonize this as well?
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        pickle_inbu = open("bu.pickle","rb")
        bu = pickle.load(pickle_inbu)
        
        pickle_inbi = open("bi.pickle","rb")
        bi = pickle.load(pickle_inbi
                         )
        pickle_inpu = open("pu.pickle","rb")
        pu = pickle.load(pickle_inpu)
        
        pickle_inqi = open("qi.pickle","rb")
        qi = pickle.load(pickle_inqi)

        if known_user and known_item:
            est = self.trainset.global_mean + bu[u] + bi[i] + np.dot(qi[i], pu[u])
        else:
                raise PredictionImpossible('User and item are unkown.')
                
        return est

    def test(self,testset,verbose = False):
        #        AlgoBase.fit(self, trainset)
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
    
            
    
    
    
    
    
    
    
    
    
    
    

        
    
    
    