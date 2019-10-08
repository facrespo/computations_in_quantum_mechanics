from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;


import quantum;

N=10;
nout=6;
nplot=4;
alpha=0.5;
m1=1;
m2=1;

H, Eig, C, E, Cplot= quantum.twodpotencial(N, nout, nplot, alpha, m1, m2);

limt=4;
dt=0.05;

psi, xx = quantum.calculatepsitwodpotencial(limt,dt,N,Cplot);

z_min, z_max = -np.real(psi).max(), np.real(psi).max()


fig1=plt.figure();
plt.pcolor(xx, xx, psi.real, cmap='RdBu', vmin=z_min, vmax=z_max)
ax=plt.axes();
plt.colorbar();
plt.xlabel('$x$', fontsize = 13);
plt.ylabel('$y$', fontsize = 13);
plt.show();
fig1.savefig('pullen-edmondspotential.jpg');

