from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;


import quantum;

N=6;
Np=N+1;
Nout=10;
epsilon=1;
v=1;
c=1;
lambdac=10000; 


H, E, C, Nav, Eord = quantum.bossehubbardimerextend(N, Np, Nout, epsilon, v, c, lambdac);

fig1=plt.figure();
#x=Nav[0:int((Np*(Np+1))/2),].real;
#iei=(Nav<=25);
#print(iei);
#y=E[0:int((Np*(Np+1))/2),].real+E[0:int((Np*(Np+1))/2),].imag;
#y=E[0:int((Np*(Np+1))/2),].real;
plt.scatter(Nav.real,E.real);
ax=plt.axes();
plt.xlabel('$<N>$', fontsize = 13);
plt.ylabel('$E_n$', fontsize = 13);
plt.show();
fig1.savefig('bossehubbardimerextend.jpg');

