import numpy as np
import scipy as sp
from variable3 import *
from readvel3 import *
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib import cm, colors
from scipy.optimize import curve_fit
font = {'size':18}
matplotlib.rcParams.update({'font.size': 14})
cmap = plt.get_cmap('Spectral_r')
rhol = 1.398
h0 = 5e-9
gamma = 0.0141
tc = np.sqrt(rhol*1000*h0**3/gamma)*1e9
fig,ax = plt.subplots(1,1)
direct = '.'
infile = direct+'/in.run'
dt = readvar(infile,'Dt')
Ly = readvar(infile,'Ly')
Lz = readvar(infile,'Lz')
dx = readvar(infile,'dx')
wmcc = readvar(infile,'wmcc')
L = readvar(infile,'L')
datafile = direct+'/data/rho-102.profile'
d1,d2 = readrow(datafile)
gasdatafile = direct+'/data/gasrho-102.profile'
d1gas,d2gas = readrow(gasdatafile)    
Time = np.array(d1['Timestep'])*dt/1e6/tc
PS = readvar(infile,'PS')
u = constants.physical_constants['atomic mass constant'][0]
m0 = 40*u
temp = readvar(infile,'T')
T = []
R_list = []

for it,t in enumerate(Time):
    print(it,t)
    x = np.array(d2['Coord1'][it])/(h0*1e10)
    rho = np.array(d2['density/mass'][it])
    A = Ly*Lz
    rhog1 = np.array(d2gas['density/mass'][it]) * A / (A - (Ly-2*wmcc)*(Lz-2*wmcc))
    rhog2 = PS*101325 / constants.k / temp * m0 * 1e-3
    if t<5:
        rhog = rhog2*(5-t)/5 + rhog1*(t/5)
    else:
        rhog = rhog1
    drho = rho-rhog
    drho[drho<0]=0
    R = np.sqrt((drho)/(rhol-rhog)*A/np.pi)/(h0*1e10)
    R_list.append(R)
    T.append(t)
#    if np.min(R)<0.5:
#        break
norm = colors.Normalize(0,T[-1])
window_size = 5
R_sum = 0
T_sum = 0
jt = 0
Time = []
Rave = []
for it,(t,R) in enumerate(zip(T,R_list)):
    Time.append(t)
    Rave.append(np.average(R))
    R_sum+=R
    T_sum+=t
    if (it+1)%window_size == 0:
        R_ave = R_sum/window_size
        t_ave = T_sum/window_size
        if jt%5==0:
            ax.plot(coarseave(x,1),coarseave(R_ave,1),color=cmap(norm(t_ave)))
        R_sum = 0
        T_sum = 0
        jt+=1
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax)
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$h$')
ax.set_title('$S=3.0,\, C=0.0049$',fontdict=font)
ax.set_ylim((0,2))
fig2,ax2 = plt.subplots(1,1)
Time = np.array(Time)
Rave = np.array(Rave)
Nave = 10
ax2.scatter(coarseave(Time,Nave),coarseave(Rave,Nave),s=40,edgecolor='b',facecolor='none',alpha=1.0)
C = 0.0054
ax2.plot(Time,Time*C+1,'k-')
def flin(x,b):
    return x*b+1
popt,pconv = curve_fit(flin,Time,Rave)
b = popt[0]
print(b)
ax2.plot(Time,Time*b+1,'b-')
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$h$')
plt.show()
