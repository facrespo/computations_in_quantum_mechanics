from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;


import quantum;

N=2;
Np=N+1;
Nout=10;
epsilon=0;
v=0.3;
c=0.6;
gamma=0.02;
dt=0.05;
maxt=2000;
nk=10;
ngraf=2000;

H, E, C, Nav, Eord, tlist, n1, n2 = quantum.openbossehubbardimer(N, Np, Nout, epsilon, v, c, gamma, dt, maxt, nk);

#fig = plt.figure();
fig, ax = plt.subplots();
ax.plot(tlist[0:ngraf],n2[0:ngraf]/N,label='$n_2(t)/N$');
ax.plot(tlist[0:ngraf],(n1[0:ngraf]+n2[0:ngraf])/N,label='$(n_1(t) + n_2(t))/N$');
ax.ylim(-0.1, 1.0)
ax=plt.axes();
ax.axis('equal')
ax.legend(loc='upper right', frameon=False);
ax.xlabel('Time t', fontsize = 13);
ax.ylabel('Relative particle number', fontsize = 13);
fig.show();
fig.savefig('openbossehubbardimer.jpg');

