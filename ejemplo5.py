from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;


import quantum;

N=6;
epsilon=1;
v=1;
c=1;


H, E, C = quantum.jordanschwinger(N, epsilon, v, c);

n=len(E);

NN=N*np.ones(n);

fig1=plt.figure();
#x=Nav[0:int((Np*(Np+1))/2),].real;
#iei=(Nav<=25);
#print(iei);
#y=E[0:int((Np*(Np+1))/2),].real+E[0:int((Np*(Np+1))/2),].imag;
#y=E[0:int((Np*(Np+1))/2),].real;
plt.scatter(NN,E.real);
ax=plt.axes();
plt.xlabel('$<N>$', fontsize = 13);
plt.ylabel('$E_n$', fontsize = 13);
plt.show();
fig1.savefig('bossehubbardimerextend.jpg');

