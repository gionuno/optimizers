# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:14:02 2016

@author: gionuno
"""

import de_solver
import numpy        as np;
import numpy.random as rd;
import matplotlib.pyplot    as plt;
import matplotlib.animation as anm;
import time;

def get_F(func_,low_,high_,N):
	x = np.linspace(low_[0],high_[0],N);
	y = np.linspace(low_[1],high_[1],N);
	z = np.zeros((N,N));
	for i in range(N):
		for j in range(N):
			z[j,i] = func_(np.asarray([x[i],y[j]]));
	return x,y,z;

def styblinski_tang(x):
	return 0.5*np.sum(x**4-16.*x**2+5.*x);

def beagle(x):
	return (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2

def holder_table(x):
	return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-np.linalg.norm(x)/np.pi)));

def cross_in_tray(x):
	return -1e-4*np.power(np.abs(np.sin(x[0])*np.sin(x[1])*np.exp(np.abs(100.-np.linalg.norm(x)/np.pi))+1.),0.1);

def rosenbrock(x):
	return 100.*(x[1]-x[0]**2)**2+(1-x[0])**2;

def ackley(x):
    D = x.shape[0];
    s = np.dot(x,x);
    t = np.sum(np.cos(2.*np.pi*x));
    return 20.+np.e-20.*np.exp(-0.2*np.sqrt(s/D))-np.exp(t/D); 
 
def langerman(x):
	a = np.asarray([3,5,2,1,7]);
	b = np.asarray([5,2,1,4,9]);
	c = np.asarray([1,2,5,2,3]);
	e = c*np.exp(-((x[0]-a)**2 + (x[1]-b)**2)/np.pi)
	f = np.cos(np.pi*((x[0]-a)**2 + (x[1]-b)**2));
	return np.dot(e,f);
	
low  = -10.*np.ones(2);
high =  10.*np.ones(2);  
P    =  100;

func = holder_table;

solver = de_solver.de_solver(func,low,high,P);

def bound(x,l,h):
	return np.mod(x-l,h-l)+l;

def mutat(x):
	return x + 1e-1*rd.randn(x.shape[0]);

def cross(x,y):
	a = rd.rand(x.shape[0]);
	return y+a*(x-y);

def delta(x_a,x_b,x_c,x_d,x_r):
	a = rd.rand(x_a.shape[0]);
	return x_r + a*(x_a-x_b) + (1-a)*(x_c-x_d);

solver.delta_ = delta;
solver.cross_ = cross;
solver.mutat_ = mutat;
solver.bound_ = bound;

X,Y,Z = get_F(lambda x: np.log(func(x)+1e2),low,high,256);

C = np.r_[np.c_[np.zeros((P,2)),np.ones((P,1))],np.c_[np.ones((P,1)),np.zeros((P,2))]];

fig  = plt.figure();
ax   = plt.axes(xlim=[low[0],high[0]],ylim=[low[1],high[1]]);
cont = ax.contourf(X,Y,Z,cmap='gray');
scat = ax.scatter(solver.X[:P,0],solver.X[:P,1],c=C,s=10);

def update(i):
	A = np.copy(solver.X[:P,:]);
	solver.iterate();
	A = np.r_[A,solver.Z];
	scat.set_offsets(A);
	return scat;

a = anm.FuncAnimation(fig,update,frames=300,interval=20);
a.save('holder_table.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
plt.show();