# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:09:43 2016

@author: gionuno
"""

import numpy        as np;
import numpy.random as rd;

class optimizer:
    def __init__(self,func_,low_,high_,P_):
        self.func_ = func_;
        
        self.low_  = low_;
        self.high_ = high_;
        
        self.P = P_;
        
        self.D = low_.shape[0];
        
        self.T = 0.;        
        
        self.X = np.dot(rd.rand(self.P+1,self.D),np.diag(self.high_-self.low_))+np.outer(np.ones(self.P+1),self.low_);
        self.Z = np.zeros((self.P,self.D));
        
        self.F = np.zeros(self.P+1);
        for p in range(self.P+1):
            self.F[p] = self.func_(self.X[p]);

    def __iterate__(self):
        raise NotImplementedError("not implemented yet");

