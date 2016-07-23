# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 02:00:58 2016

@author: gionuno
"""

import optimizers;
import numpy        as np;
import numpy.random as rd;

class sa_solver(optimizers.optimizer):				
    def iterate(self):
        #X_r = np.copy(self.X[-1,:]);
        for p in range(self.P):
            X_p = np.copy(self.X[p,:]);
            Y_p = (self.high_-self.low_)*rd.rand(self.D)+self.low_;
            self.Z[p,:] = self.bound_(self.mutat_(self.cross_(X_p,Y_p)),self.low_,self.high_);
            F_z = self.func_(self.Z[p,:]);
            e_T = -(F_z-self.F[p])/(0.01*self.T+1.);
            if F_z < self.F[p] or np.log(rd.rand()) < e_T:
                self.F[p] = F_z;
                self.X[p,:] = np.copy(self.Z[p,:]);
        r_ = -1;
        for p in range(self.P):
            if self.F[p] < self.F[-1]:
                self.F[-1] = self.F[p];
                r_ = p;
        if r_ >= 0:
            self.X[-1,:] = np.copy(self.X[r_,:]);
        self.T += 1.;
