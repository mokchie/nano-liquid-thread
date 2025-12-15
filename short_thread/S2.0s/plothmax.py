import numpy as np
import scipy as sp
from variable3 import *
from readvel3 import *
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import curve_fit
font = {'size':11}
matplotlib.rcParams.update({'font.size': 13})
cmap = plt.get_cmap('spring')
fig1,ax1 = plt.subplots(1,1)
infile = 'in.run'
wmcc = readvar(infile,'wmcc')
dt = readvar(infile,'Dt')
Ly = readvar(infile,'Ly')
Lz = readvar(infile,'Lz')
L = readvar(infile,'L')
Fn = readvar(infile,'fn')
T = readvar(infile,'T')

PS = readvar(infile,'PS')
R0 = readvar(infile,'R0')
tot = readvar(infile,'tot')
u = constants.physical_constants['atomic mass constant'][0]
m0 = 40*u

Rmin_all = []
Rmax_all = []
Rave_all = []
fp =  open('hmax_md.npy','wb')
np.save(fp,len(Fn))
for cn,fn in enumerate(Fn):
    print(fn)
    datafile = 'data/rho-%d.profile'%fn
    d1,d2 = readrow(datafile)
    gasdatafile = 'data/gasrho-%d.profile'%fn
    d1gas,d2gas = readrow(gasdatafile)        
    Time = np.array(d1['Timestep'])*dt
    rhol = 1.398
    #rhog = PS*101325 / constants.k / T * m0 * 1e-3

    Tt = []
    Rmin = []
    Rmax = []
    Rave = []
    pnum = []
    for it,t in enumerate(Time):
        x = np.array(d2['Coord1'][it])
        rho = np.array(d2['density/mass'][it])
        A = Ly*Lz
        rhog = np.array(d2gas['density/mass'][it]) * A / (A - (Ly-2*wmcc)*(Lz-2*wmcc))        
        x  = np.array(coarseave(x,1))
        rho = np.array(coarseave(rho,1))
        rhog = np.array(coarseave(rhog,1))
        pnum.append(np.sum(d2['Ncount'][it]))

        drho = rho-rhog
        drho[drho<0]=0
        R = np.sqrt((drho)/(rhol-rhog)*A/np.pi)
        rave = np.average(R)
        Rmin.append(np.min(R))
        Rmax.append(np.max(R))
        Rave.append(rave)
        Tt.append(t)
        if np.min(R)<5:
            break
    Tt = np.array(Tt)/1e6
    Rmin = np.array(Rmin)
    Rmax = np.array(Rmax)    
    Rave = np.array(Rave)
    pnum = np.array(pnum)
    Rmin_all.append(Rmin)
    Rmax_all.append(Rmax)    
    Rave_all.append(Rave)
    
    ax1.scatter(Tt,(Rmax-Rmin)/2/R0,s=80,edgecolor=cmap(cn/len(Fn)),facecolor='none',alpha=0.2)
    np.save(fp,Tt)
    np.save(fp,np.abs((Rmax-Rmin)/2/R0))
fp.close()
Hmax_list = [[] for i in range(len(Time))]
Hmax_sq_list = [[] for i in range(len(Time))]
Rave_list = [[] for i in range(len(Time))]
for Rmin,Rmax,Rave in zip(Rmin_all,Rmax_all,Rave_all):
    for i,(rmin,rmax,rave) in enumerate(zip(Rmin,Rmax,Rave)):
        Hmax_sq_list[i].append(((rmax-rmin)/2/R0)**2)
        Hmax_list[i].append(((rmax-rmin)/2/R0))        
        Rave_list[i].append(rave)

hmax_sq_ave = []
hmax_sq_std = []
for hmax_sq_list in Hmax_sq_list:
    hmax_sq_ave.append(np.average(hmax_sq_list))
    hmax_sq_std.append(np.std(hmax_sq_list))
hmax_ave = []
hmax_std = []    
for hmax_list in Hmax_list:
    hmax_ave.append(np.average(hmax_list))
    hmax_std.append(np.std(hmax_list))        
Rave_ave = []
for rave in Rave_list:
    Rave_ave.append(np.average(rave))
Rave_ave = np.array(Rave_ave)
hmax_sq_ave = np.array(hmax_sq_ave)
hmax_sq_std = np.array(hmax_sq_std)
ax1.plot(Time/1e6,np.sqrt(hmax_sq_ave),'k--')
with open('hmax_ave_md.npy','wb') as fp:
    np.save(fp,Time/1e6)
    np.save(fp,np.sqrt(hmax_sq_ave))
    np.save(fp,hmax_sq_std)
    np.save(fp,Rave_ave)
ax1.set_ylim((0,1.0))
ax1.set_xlabel(r'$t (ns)$')
ax1.set_ylabel(r'$\tilde{h}$')
ax1.set_xlabel(r'$t (ns)$')
ax1.set_ylabel(r'$\tilde{h}_{max}$')


plt.show()
