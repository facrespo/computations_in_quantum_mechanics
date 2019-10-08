from __future__ import print_function 

import numpy as np;
import scipy.linalg as scl;
import matplotlib;
import matplotlib.pyplot as plt;


import quantum;

def V(x,x0,N):
    v=x-x0*np.identity(N);
    v=(1/2)*np.dot(v,v);
    return v;


#N=4;
#a=quantum.generar_a(N);
#n=np.zeros((N+1,1));
#n[N-2,0]=1;
#a1=np.dot(a,n);
#a2=np.dot(a.transpose(),n);
#a3=np.dot(a.transpose(),np.dot(a,n));


N=100;
x0=2.5;
Nite=100;
x=quantum.generar_x(N);
p=quantum.generar_p(N);
H=(1/2)*np.dot(p,p)+V(x,x0,N);

E, V=scl.eig(H);
E1=np.sort(E);

psi=V[0:N,0];
tstep=20;
tfin=(Nite-1)*(tstep);
t=np.zeros((Nite,1));
xav=np.complex(0,0)*np.zeros((Nite,1));
xav[0]=np.dot(psi.transpose(),np.dot(x,psi));
U=scl.expm(-1j*H*tstep);

xav, t = quantum.dinamical_cal(t, psi, tstep, x, U, xav, Nite);
plt.plot(t, xav);
plt.plot(t, x0*(np.cos((E[2]-E[1])*t)),'bo');
plt.xlabel('t');
plt.ylabel('<x>');
plt.title('Wave');
plt.show();

d=2*np.pi;
F=0.005;
Delta=1;
nmax=60;
nmin=-60;

Hm1, nn1, nt1 = quantum.Hneh(d, F, Delta, nmax, nmin);

J=80;
NB=2; # N Bloch periods

Psi1, tt1 = quantum.Timeevolution(Hm1, nn1, nt1, J, NB, d, F, nmax, nmin);
Psi1 = np.around(Psi1, decimals=5);

fig1=plt.figure();
#plt.imshow(abs(Psi), cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1);
plt.imshow(abs(Psi1));
ax=plt.axes();
plt.colorbar();
plt.xlabel('$t/T_B$', fontsize = 13);
plt.ylabel('<x>', fontsize = 13);
ax.xaxis.set_ticklabels([' ', '0', ' ', ' ', '1',' ',' ','2']);
ax.yaxis.set_ticklabels(['60', '40', '20', '0', '-20','-40','-60']);
plt.show();
fig1.savefig('plot1.jpg');


sig=0.005;

Psi2, tt2 = quantum.TimeevolutionG(Hm1, nn1, nt1, J, NB, d, F, nmax, nmin, sig);


fig1=plt.figure();
plt.imshow(abs(Psi2));
ax=plt.axes();
plt.colorbar();
plt.xlabel('$t/T_B$', fontsize = 13);
plt.ylabel('n', fontsize = 13);
ax.xaxis.set_ticklabels([' ', '0', ' ', ' ', '1',' ',' ','2']);
ax.yaxis.set_ticklabels(['60', '40', '20', '0', '-20','-40','-60']);
plt.show();
fig1.savefig('plot2.jpg');


nmax=160;
nmin=-40;

Hp, nn1, nt1 = quantum.Hneh(d, F, Delta, nmax, nmin);

Hm, nn1, nt1 = quantum.Hneh(-1*d, F, Delta, nmax, nmin);


Psi3, tt3 = quantum.TimeevolutionFlipFlow(Hp, Hm, nn1, nt1, J, NB, d, F, nmax, nmin, sig);

fig1=plt.figure();
plt.imshow(abs(Psi3));
ax=plt.axes();
plt.colorbar();
plt.xlabel('$t/T_B$', fontsize = 13);
plt.ylabel('n', fontsize = 13);
ax.xaxis.set_ticklabels([' ','0', ' ', ' ','2']);
ax.yaxis.set_ticklabels([' ', '-20', '0', '20','40','60','80','100','120','140']);
plt.show();
fig1.savefig('plot3.jpg');

jj=1000;

Jp, Jm, Jx, Jy, Jz = quantum.generar_Jp(jj);

Ix=1/3;
Iy=1/2;
Iz=1;

HJ=quantum.generar_JH(Jx,Jy,Jz,Ix,Iy,Iz);
Ev, Evec=(scl.eig(HJ));
#E=np.sort(Ev);
En=Ev/(jj**2);


fig1=plt.figure();
plt.hist(En,bins=50);
plt.xlabel('$E/j^2$', fontsize = 13);
plt.ylabel('$\Delta N/ \Delta E$', fontsize = 13);
plt.show();
fig1.savefig('histogramamomentun.jpg');