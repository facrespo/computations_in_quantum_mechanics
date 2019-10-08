from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;
from scipy import sparse;

import quantum;

ND=(3*10)+3;
N=ND-1;
K=1;
Phi=0.8*np.pi;
u=0.5;
U=(K*u)/N;
epsilon=10**(-6);

H_N, C1, Eigd, C1, dH_dPhi_N, Jav = quantum.bossehubbartrimer(ND, N, K, Phi, U, u, epsilon);

fig1=plt.figure();
#iei=(Nav<=25);
#print(iei);
#y=E[0:int((Np*(Np+1))/2),].real+E[0:int((Np*(Np+1))/2),].imag;
plt.scatter(Jav,Eigd);
ax=plt.axes();
plt.xlabel('$<J>$', fontsize = 20);
plt.ylabel('$E_n/u$', fontsize = 13);
plt.show();
fig1.savefig('bossehubbartrimer.jpg');

