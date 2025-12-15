import numpy as np
import scipy as sp
from variable3 import *
from readvel3 import *
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
import scipy.fft as fft
font = {'size':11}
matplotlib.rcParams.update({'font.size': 13})
cmap = plt.get_cmap('jet')
fig1,ax1 = plt.subplots(1,1)
fig2,ax2 = plt.subplots(1,1)
colors = ['C%s'%i for i in range(10)]
direct = '.'
infile = direct+'/in.run'
dt = readvar(infile,'Dt')
Ly = readvar(infile,'Ly')
Lz = readvar(infile,'Lz')
wmcc = readvar(infile,'wmcc')
L = readvar(infile,'L')
PS = readvar(infile,'PS')
R0 = readvar(infile,'R0')
dt = readvar(infile,'Dt')
tot = readvar(infile,'tot')
SP_list = []
print('R0 =',R0)
T = readvar(infile,'T')
u = constants.physical_constants['atomic mass constant'][0]
m0 = 40*u
Fn = readvar(infile,'fn')
for fn in Fn:
    print(fn)
    datafile = direct+'/data/rho-%d.profile'%fn
    d1,d2 = readrow(datafile)
    gasdatafile = direct+'/data/gasrho-%d.profile'%fn
    d1gas,d2gas = readrow(gasdatafile)    
    Time = np.array(d1['Timestep'])*dt
    rhol = 1.398
    #rhog = PS*101325 / constants.k / T * m0 * 1e-3
    #print('rhog =',rhog)
    Tl = []
    Rmin = []
    nw = 20
    SP = []
    for it,t in enumerate(Time):
        pass
        x = np.array(d2['Coord1'][it])
        dx = x[2]-x[1]        
        rho = np.array(d2['density/mass'][it])
        #rhog = np.array(d2gas['density/mass'][it])
        A = Ly*Lz        
        rhog = np.array(d2gas['density/mass'][it])/(A - (Ly-2*wmcc)*(Lz-2*wmcc))*A        
        drho = rho-rhog
        #drho[drho<0]=0
        R = np.sqrt((drho)/(rhol-rhog)*A/np.pi)
        Sp = fft.fft((R-np.sqrt(np.average(R**2))))[1:nw]*dx/R0**2
        #Sp = fft.fft((R-np.average(R)))[1:nw]*dx/R0**2
        SP.append(Sp)
        K = 2*np.pi*fft.fftfreq(len(R),d=dx)[1:nw]
        Rmin.append(np.min(R))    
        Tl.append(t)
        #print(t,np.average(R))
        #if t>tot*dt:
        #    break
    SP_list.append(np.transpose(np.array(SP)))
    Tl = np.array(Tl)/1e6
SP_list = np.array(SP_list)
SP_ave = np.sqrt(np.average(np.abs(SP_list)**2,axis=0))

SP_std = np.std(np.abs(SP_list),axis=0)

for i in range(1,nw):
    if i%2==0:
        ax2.plot(Tl,SP_ave[i,:],color=cmap(i/nw),label='k=%.2f'%(K[i]*R0))
        ax2.fill_between(Tl,(SP_ave[i,:]-SP_std[i,:]/2),(SP_ave[i,:]+SP_std[i,:]/2),color=cmap(i/nw),alpha=0.2)

Ns = 40
for i in range(len(Tl)):
    if i%Ns==0:
        ax1.plot(K[:]*R0,SP_ave[:,i],color=cmap(i/len(Tl)),label='t=%.2f'%(Tl[i]))
        print(np.sqrt(np.trapz((SP_ave[:,i]*R0)**2,K)/np.pi/L)*R0)
        pass
print('')
nu = 1.77e-7
T = readvar(infile,'T')
kB = constants.k
rho0 = 1398
gamma = 0.0141
h0 = readvar(infile,'R0')*1e-10
Oh = nu*np.sqrt(rho0/gamma/h0)
print('Oh =',Oh)
Th = sqrt(kB*T/gamma)/h0
print('Th =',Th)
t_star = np.sqrt(rho0*h0**3/gamma)
Tm = Tl-Tl[0]
TT = []
H_list = []
for i in range(len(Tm)):
    if i%Ns==0:
        t = Tm[i]*1e-9/t_star
        TT.append(t)
        print(Tm[i],t)
        alpha = 3*Oh*(K*R0)**2
        beta = np.sqrt((9*Oh**2-2)*(K*R0)**4+2*(K*R0)**2)
        HLE_sq = (SP_ave[:,0])**2 * ((np.exp((-alpha+beta)*t/2)+np.exp(-(alpha+beta)*t/2))/2 + alpha/beta * (np.exp((-alpha+beta)*t/2)-np.exp(-(alpha+beta)*t/2))/2)**2
        #HLE_sq = (SP_ave[:,0]/R0**2)**2 * np.exp(-alpha*t)*(np.cosh(beta*t/2)+alpha/beta*np.sinh(beta*t/2))**2
        HFLUC_sq = 3*(L/R0)*Oh/np.pi*Th**2 * (K*R0)**4 * ((alpha**2-beta**2)*np.exp(-alpha*t)-alpha**2* (np.exp((-alpha+beta)*t)+np.exp(-(alpha+beta)*t))/2 - alpha*beta*(np.exp((-alpha+beta)*t) - np.exp(-(alpha+beta)*t))/2 + beta**2) / (alpha*beta**2*(alpha**2-beta**2))
        #HFLUC_sq = 3*L/R0*Oh/np.pi*Th**2*(K*R0)**4 * (-alpha**2*np.cosh(beta*t)-alpha*beta*np.sinh(beta*t)+alpha**2-beta**2+beta**2*np.exp(alpha*t))/(alpha*beta**2*(alpha**2-beta**2)*np.exp(alpha*t))
        H = np.sqrt(HLE_sq+HFLUC_sq)
        ax1.plot(K*R0,H,linestyle='--',color=cmap(i/len(Tm)))
        H_list.append(H)
        print(np.sqrt(np.trapz((H*R0)**2,K)/np.pi/L)*R0)
TT = np.array(TT)
H_list = np.array(H_list)
with open('md_result.npy','wb') as fp:
    np.save(fp,TT)
    np.save(fp,K*R0)
    np.save(fp,SP_ave[:,0::Ns])
    np.save(fp,H_list)
print(TT)
print(K*R0)

ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'spectrum')
ax2.set_xlabel(r'$T (ns)$')
ax2.set_ylabel(r'spectrum')
ax2.legend(ncol=3)
plt.show()
