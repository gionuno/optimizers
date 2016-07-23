# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 20:28:40 2016

@author: gionuno
"""

import optimizers;
import numpy        as np;
import numpy.random as rd;

class de_solver(optimizers.optimizer):				
    def iterate(self):
        X_r = np.copy(self.X[-1,:]);
        for p in range(self.P):
            a = rd.randint(0,self.P);
            b = rd.randint(0,self.P);
            c = rd.randint(0,self.P);	
            d = rd.randint(0,self.P);			
            while a == p:
                a = rd.randint(0,self.P);
            while b == p or b == a:
                b = rd.randint(0,self.P);
            while c == p or c == b or c == a:
                c = rd.randint(0,self.P);
            while d == p or d == c or d == b or c == a:
                d = rd.randint(0,self.P);
            X_p = np.copy(self.X[p,:]);
            X_a = np.copy(self.X[a,:]);
            X_b = np.copy(self.X[b,:]);
            X_c = np.copy(self.X[c,:]);
            X_d = np.copy(self.X[d,:]);
            
            Y_p = self.delta_(X_a,X_b,X_c,X_d,X_r);
            self.Z[p,:] = self.bound_(self.mutat_(self.cross_(X_p,Y_p)),self.low_,self.high_);
            F_z = self.func_(self.Z[p,:]);
            if F_z < self.F[p]:
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
