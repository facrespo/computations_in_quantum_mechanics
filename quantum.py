import numpy as np;
import numpy.linalg as la;
import scipy.stats;
from scipy import stats, special, optimize, sparse;
from scipy.stats import norm, mvn, distributions;
import scipy.linalg as scl;
import scipy;
import math;

tol = 1e-12;

def generar_a(N):
    a = np.diagflat(np.sqrt(np.arange(1,N+1)), 1);
    return a;

def generar_x(N):
    x = np.complex(1,0)*((1/np.sqrt(2))*(np.diag(np.sqrt(np.arange(1,N)), -1)+np.diag(np.sqrt(np.arange(1,N)), 1)));
    return x;

def generar_p(N):
    p = (np.complex(0,1)/np.sqrt(2))*(np.diag(np.sqrt(np.arange(1,N)), -1)-np.diag(np.sqrt(np.arange(1,N)), 1));
    return p;

def dinamical_cal(t, psi, tstep, x, U, xav, nstep):
    for nt in range(0,nstep-1):
        t[nt+1] = t[nt] + tstep;
        psi=np.dot(U,psi);
        xav[nt+1]=np.matmul(psi.transpose(),np.matmul(x,psi));
    return xav, t;

def Hneh(d, F, Delta, nmax, nmin):
    nt=np.arange(nmin,nmax+1);
    nn=len(nt);
    m=np.ones((1,nn-1));
    H=d*F*np.diag(nt,0)+(Delta/4)*(np.diag(m.ravel(),-1)+np.diag(m.ravel(),1));
    return H, nn, nt;

def Timeevolution(H, nn, nt, J, NB, d, F, nmax, nmin):
    psi = 0*nt;
    psi[-1*nmin]=1;
    Psi=np.complex(0,0)*np.zeros((nn, NB*J+1));
    Psi[:,0]=psi;
    U=scl.expm((-1j*H*2*np.pi)/(d*F*J));
    for k in range(0,NB*J): 
        Psi[:,k+1]=np.matmul(U,Psi[:,k]);
    t=np.arange(0,NB*J+1);
    t=t/J;
    return Psi, t;


def TimeevolutionG(H, nn, nt, J, NB, d, F, nmax, nmin, sig):
    phi0=np.pi/2;
    psi = np.exp((-sig*(nt**2))+np.complex(0,1)*nt*phi0);
    psi=psi/np.sqrt(sum((abs(psi))**2));
    Psi=np.complex(0,0)*np.zeros((nn, NB*J+1));
    Psi[:,0]=psi;
    U=scl.expm((-1j*H*2*np.pi)/(d*F*J));
    for k in range(0,NB*J): 
        Psi[:,k+1]=np.matmul(U,Psi[:,k]);
    t=np.arange(0,NB*J+1);
    t=t/J;
    return Psi, t;

def TimeevolutionFlipFlow(Hp, Hm, nn, nt, J, NB, d, F, nmax, nmin, sig):
    phi0=np.pi/2;
    #psi =np.exp(-sig*(nt**2)+np.complex(0,1)*nt*phi0);
    psi=np.exp((-sig*(nt**2))+np.complex(0,1)*nt*phi0);
    psi=psi/np.sqrt(sum((abs(psi))**2));
    psi.real[abs(psi.real) < tol] = 0.0;
    psi.imag[abs(psi.imag) < tol] = 0.0;
    psi=np.conj(psi);
    Up=scl.expm((-1j*Hp*2*np.pi)/(d*F*J));
    Um=scl.expm((-1j*Hm*2*np.pi)/(d*F*J));
    Psi=np.complex(0,0)*np.zeros((nn, NB*J+1));
    Psi[:,0]=psi;
    nn1=0;
    for i in range(0,2*NB):
       for k in range(0,int(J/4)): 
           Psi[:,nn1+1]=np.matmul(Up,Psi[:,nn1]);
           nn1 = nn1+1;
       for k in range(0,int(J/4)): 
           Psi[:,nn1+1]=np.matmul(Um,Psi[:,nn1]);    
           nn1 = nn1+1;
    t=np.arange(0,NB*J+1);
    t=t/J;
    return Psi, t;

def generar_Jp(jj):
    mm=np.arange(-1*jj,jj);
    Jp = np.diag(np.sqrt(jj*(jj+1)-mm*(mm+1)), 1);
    Jm=Jp.transpose();
    Jx=(Jm+Jp)/2;
    Jy=-1j*(Jm-Jp)/2;
    Jz=(np.matmul(Jp,Jm)-np.matmul(Jm,Jp))/2;
    return Jp, Jm, Jx, Jy, Jz;

def generar_JH(Jx,Jy,Jz,Ix,Iy,Iz):
    H = (np.matmul(Jx,Jx)/(2*Ix))+(np.matmul(Jy,Jy)/(2*Iy))+(np.matmul(Jz,Jz)/(2*Iz));
    return H;

def V(x1,x2,alpha):
    v=(1/2)*np.matmul(x1,x1)+(1/2)*np.matmul(x2,x2)+alpha*(np.matmul(np.matmul(x1,x1),np.matmul(x2,x2)))
    return v;

def twodpotencial(N, nout, nplot, alpha, m1, m2):
    m=np.sqrt(np.arange(1,N));
    md=np.diag(m.ravel(),-1);
    x= generar_x(N);
    p= generar_p(N);
    I = np.identity(N);
    x1=np.kron(x,I);
    x2=np.kron(I,x);
    p1=np.kron(p,I);
    p2=np.kron(I,p);
    H=(1/(2*m1))*np.matmul(p1,p1)+(1/(2*m2))*np.matmul(p2,p2)+ V(x1,x2,alpha);
    Eig, C=scl.eig(H);
    Eig2=np.sort(Eig, kind='mergesort');
    Eig2=Eig2[0:nout];
    #ieig=orden(Eig, Eig2);
    ieig=np.argsort(Eig)[0:nout];
    C2=C[:,ieig];
    E=np.diag(Eig2);
    Cplot=np.reshape(C2[:,nplot-1],(N,N));
    return H, Eig, C, E, Cplot;

#def orden(E1, E2):
#    n=E2.shape;
#    a=list(range(0,n[0]));
#    for i in range(0,n[0]):
#        a[i]=int(0);
#        aa=np.where(E1==E2[i]);
#        a[i]=int(aa[0][0]);
#    return a;

def calculatepsitwodpotencial(limt,dt,N,Cplot):
    xx=np.arange(-1*limt,limt+dt,dt);
    Nx=xx.shape[0];
    h0=[1];
    hermval=np.zeros((Nx,N));
    xx2=xx**2;
    xx2a=-0.5*xx2
    exx2a=np.exp(xx2a);
    hermval[:,0]=np.polyval(h0,xx)*exx2a;
    h1=[np.sqrt(2),0];
    hermval[:,1]=np.polyval(h1,xx)*exx2a;
    v1=[1,0];
    v0=[0,0,1];
    for i in range(2,N):
        h2=np.sqrt(2/i)*np.convolve(h1,v1)-np.sqrt(1-(1/i))*np.convolve(h0,v0);
        h0=h1;
        h1=h2;
        hermval[:,i]=np.polyval(h2,xx)*exx2a;
    psi=np.matmul(hermval,np.matmul(Cplot,hermval.transpose()));   
    return psi, xx;

def bossehubbardimer(N, Np, Nout, epsilon, v, c):
    m=np.sqrt(np.arange(1,N+1));
    a=np.diag(m.ravel(),1);
    ad=a.transpose();
    I = np.identity(Np);
    a1=np.kron(a,I);
    ad1=a1.transpose();
    a2=np.kron(I,a);
    ad2=a2.transpose();
    H=(epsilon)*(np.matmul(ad1,a1)-np.matmul(ad2,a2)) + (v)*(np.matmul(ad1,a2)+np.matmul(ad2,a1)) + (c/2)*np.matmul(np.matmul(ad1,a1)-np.matmul(ad2,a2),np.matmul(ad1,a1)-np.matmul(ad2,a2));
    Eig, C=scl.eig(H);
    Eig2=np.sort(Eig.real, kind='mergesort');
    ieig2=np.argsort(Eig.real);
    C1=C[:,ieig2];
    Nav=np.matmul(np.conj(C1.transpose()),np.matmul(np.matmul(ad1,a1)+np.matmul(ad2,a2),C1));
    Nav=np.diag(Nav);
    Eig3=np.sort(Nav.real, kind='mergesort');
    ieig=np.argsort(Nav.real);
    Eigord=Eig2[ieig,];
    return H, Eig2, C1, Nav, Eigord;

def bossehubbardimerextend(N, Np, Nout, epsilon, v, c, lambdac):
    m=np.sqrt(np.arange(1,N+1));
    a=np.diag(m.ravel(),1);
    ad=a.transpose();
    I = np.identity(Np);
    a1=np.kron(a,I);
    ad1=a1.transpose();
    a2=np.kron(I,a);
    ad2=a2.transpose();
    H=(epsilon)*(np.matmul(ad1,a1)-np.matmul(ad2,a2)) + (v)*(np.matmul(ad1,a2)+np.matmul(ad2,a1)) + (c/2)*np.matmul(np.matmul(ad1,a1)-np.matmul(ad2,a2),np.matmul(ad1,a1)-np.matmul(ad2,a2));
    H1=H-(lambdac)*(np.matmul(ad1,a1)+np.matmul(ad2,a2)-N*np.kron(I,I));
    Eig, C=scl.eig(H1);
    Eig2=np.sort(Eig.real, kind='mergesort');
    ieig2=np.argsort(Eig.real);
    C1=C[:,ieig2];
    #Filtering
    ieig3=(abs(Eig2) < 10*epsilon*N);
    Eig3=Eig2[ieig3];
    C2=C1[:,ieig3];
    #N_op = np.matmul(ad1,a1)+np.matmul(ad2,a2);
    #iN=(abs(np.diag(N_op)-N) < 10e-6);
    #N_p=N_op[:,iN]/N;
    #H_N = np.matmul(N_p.transpose(),np.matmul(H,N_p));
    #Eig4, C4=scl.eig(H1);
    Nav=np.matmul(np.conj(C2.transpose()),np.matmul(np.matmul(ad1,a1)+np.matmul(ad2,a2),C2));
    Nav=np.diag(Nav);
    Eig4=np.sort(Nav.real, kind='mergesort');
    ieig=np.argsort(Nav.real);
    Eigord=Eig3[ieig,];
    return H1, Eig3, C2, Nav, Eigord;

def jordanschwinger(N, epsilon, v, c):
    jj=N/2;
    Jp, Jm, Jx, Jy, Jz = generar_Jp(jj);
    H = 2*epsilon*Jz+2*v*Jx+2*c*np.matmul(Jz,Jz);
    Eig, C=scl.eig(H);
    Eig2=np.sort(Eig.real, kind='mergesort');
    ieig2=np.argsort(Eig.real);
    C1=C[:,ieig2];
    return H, Eig2, C1;

def bossehubbartrimer(ND, N, K, Phi, U, u, epsilon):
    m=np.sqrt(np.arange(1,N+1));
    a=scipy.sparse.dia_matrix(scipy.sparse.diags(m.ravel(),1));
    ad=a.transpose();
    I = scipy.sparse.eye(ND);
    a1=scipy.sparse.kron(scipy.sparse.kron(a,I),I);
    ad1=a1.transpose();
    a2=scipy.sparse.kron(scipy.sparse.kron(I,a),I);
    ad2=a2.transpose();
    a3=scipy.sparse.kron(scipy.sparse.kron(I,I),a);
    ad3=a3.transpose();
    N_op=((ad1@a1+ad2@a2)+ad3@a3);
    iN=(abs(N_op.diagonal()-N) < epsilon);
    N_p = scipy.sparse.dia_matrix(N_op[:,iN]/N);
    H=(-K/2)*(np.exp((1j*Phi)/3)*((ad2@a1+ad3@a2)+ad1@a3)+np.exp((-1j*Phi)/3)*((ad1@a2+ad3@a1)+ad2@a3))+(U/2)*((((ad1@ad1)*(a1@a1))+((ad2@ad2)*(a2@a2)))+((ad3@ad3)*(a3@a3)));
    H_N=(N_p.transpose()@H)@N_p;
    n1=H_N.get_shape();
    Eig, C=scipy.sparse.linalg.eigs(H_N,k=n1[0]-3,which='SR');
    Eigd=Eig/u;
    Ev=Eigd.real;
    Eig2=np.sort(Ev, kind='mergesort');
    ieig2=np.argsort(Ev);
    C1=C[:,ieig2];
    dH_dPhi=(-K/2)*(((1j)/3)*np.exp((1j*Phi)/3)*((ad2@a1+ad3@a2)+ad1@a3)+((-1j)/3)*np.exp((-1j*Phi)/3)*((ad1@a2+ad3@a1)+ad2@a3));
    dH_dPhi_N=(N_p.transpose()@dH_dPhi)@N_p;
    Jav=((((C1.transpose()).conjugate())@dH_dPhi_N)@C1);
    Jav=np.diag(Jav.real);
    return H_N, C1, Eig2, C1, dH_dPhi_N, Jav;

def openbossehubbardimer(N, Np, Nout, epsilon, v, c, gamma,  dt, maxt, nk):
    m=np.sqrt(np.arange(1,N+1));
    a=np.diag(m.ravel(),1);
    ad=a.transpose();
    I = np.identity(Np);
    a1=np.kron(a,I);
    ad1=a1.transpose();
    a2=np.kron(I,a);
    ad2=a2.transpose();
    H=(epsilon)*(np.matmul(ad1,a1)-np.matmul(ad2,a2)) + (v)*(np.matmul(ad1,a2)+np.matmul(ad2,a1)) + (c/2)*np.matmul(np.matmul(ad1,a1)-np.matmul(ad2,a2),np.matmul(ad1,a1)-np.matmul(ad2,a2));
    psi0_1=np.zeros((Np,1));
    psi0_1[0]=1;
    psi0_2=np.zeros((Np,1));
    psi0_2[N]=1;
    psi0=np.kron(psi0_1,psi0_2);
    rho=np.matmul(psi0,psi0.transpose());
    tlist=np.arange(0,maxt,dt);
    dim1=tlist.shape[0];
    n1=np.zeros((dim1,1));
    n2=np.zeros((dim1,1));
    for l in range(0,dim1):
        n2[l]=np.matmul(rho,np.matmul(ad2,a2)).trace();
        n1[l]=np.matmul(rho,np.matmul(ad1,a1)).trace();
        for k in range(1,nk):
            rho_pred=rho-1j*(dt/nk)*(np.matmul(H,rho)-np.matmul(rho,H))+0.5*gamma*(dt/nk)*(np.matmul(a2,np.matmul(rho,ad2))-np.matmul(ad2,np.matmul(a2,rho))+np.matmul(a2,np.matmul(rho,ad2))-np.matmul(rho,np.matmul(ad2,a2)));
            rho_m=0.5*(rho+rho_pred);
            rho=rho-1j*(dt/nk)*(np.matmul(H,rho_m)-np.matmul(rho_m,H))+0.5*gamma*(dt/nk)*(np.matmul(a2,np.matmul(rho_m,ad2))-np.matmul(ad2,np.matmul(a2,rho_m))+np.matmul(a2,np.matmul(rho_m,ad2))-np.matmul(rho_m,np.matmul(ad2,a2)));
    Eig, C=scl.eig(H);
    Eig2=np.sort(Eig.real, kind='mergesort');
    ieig2=np.argsort(Eig.real);
    C1=C[:,ieig2];
    Nav=np.matmul(np.conj(C1.transpose()),np.matmul(np.matmul(ad1,a1)+np.matmul(ad2,a2),C1));
    Nav=np.diag(Nav);
    Eig3=np.sort(Nav.real, kind='mergesort');
    ieig=np.argsort(Nav.real);
    Eigord=Eig2[ieig,];
    return H, Eig2, C1, Nav, Eigord, tlist, n1, n2;