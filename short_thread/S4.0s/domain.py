from create_atom import *
from variable3 import *
import numpy as np
from scipy import constants
eps = 1e-6
S = 4.0
R = 50.0
T = 84.09
Na = constants.N_A
u = constants.physical_constants['atomic mass constant'][0]
m0 = 40*u
KR = np.exp(1.42e-2*m0/1398/constants.k/T/(R*1e-10))
print('KR =',KR)
P0 = Na/(40e-3/3.22) * constants.k * T / 101325
print('P0 =',P0)

liquid_data_file = 'rho1398.data'
print('using liquid data file ',liquid_data_file,' ...')
gas_data_file = 'rho12.88.data'
liquid_sample_positions = []
liquid_sample_velocities = []
with open(liquid_data_file,'r') as fp:
    for i in range(2):
        fp.readline()
    line = fp.readline()
    num = int(line.strip().split()[0])
    for i in range(2):
        fp.readline()    
    line = fp.readline()
    line_items = line.strip().split()
    liquid_sample_dx = float(line_items[1])-float(line_items[0])
    for i in range(13):
        fp.readline()
    for i in range(num):
        line = fp.readline()
        position = [float(item) for item in line.strip().split()[4:7]]
        liquid_sample_positions.append(position)
    for i in range(3):
        fp.readline()
    for i in range(num):
        line = fp.readline()
        vel = [float(item) for item in line.strip().split()[1:4]]
        liquid_sample_velocities.append(vel)
gas_sample_positions = []
gas_sample_velocities = []

with open(gas_data_file,'r') as fp:
    for i in range(2):
        fp.readline()
    line = fp.readline()
    num = int(line.strip().split()[0])
    for i in range(2):
        fp.readline()    
    line = fp.readline()
    line_items = line.strip().split()
    gas_sample_dx = float(line_items[1])-float(line_items[0])
    for i in range(13):
        fp.readline()
    for i in range(num):
        line = fp.readline()
        position = [float(item) for item in line.strip().split()[4:7]]
        gas_sample_positions.append(position)
    for i in range(3):
        fp.readline()
    for i in range(num):
        line = fp.readline()
        vel = [float(item) for item in line.strip().split()[1:4]]
        gas_sample_velocities.append(vel)
liquid_sample_positions = np.array(liquid_sample_positions)
gas_sample_positions = np.array(gas_sample_positions)
liquid_sample_velocities = np.array(liquid_sample_velocities)
gas_sample_velocities = np.array(gas_sample_velocities)

if np.abs(liquid_sample_dx - gas_sample_dx)>eps:
    print('Error! The width of the gas sample and liquid sample are not consistent!')
    exit(0)
sample_dx = liquid_sample_dx

L =  np.round((R*2*np.pi/0.9)/sample_dx)*sample_dx
print('k=',2*np.pi/(L/R))
Lx = L
Ly = 500
Lz = 500
xlo = -Lx/2
xhi = Lx/2
ylo = -Ly/2
yhi = Ly/2
zlo = -Lz/2
zhi = Lz/2
rc = 15.0

liquid_region = cylinder(make_point(-L/2,0,0),make_vect(L,0,0),R)
gas_region = sub_region(box((xlo,ylo,zlo),(xhi,yhi,zhi)),cylinder(make_point(-L/2,0,0),make_vect(L,0,0),R+2.0))

num_of_liquid_samples = int(np.round(Lx/liquid_sample_dx))
liquid_positions = np.concatenate([liquid_sample_positions - np.array([(-liquid_sample_dx/2-xlo)-i*liquid_sample_dx,0,0]) for i in range(num_of_liquid_samples)])
liquid_velocities = np.concatenate([liquid_sample_velocities for i in range(num_of_liquid_samples)])
liquid_mask = np.array([liquid_region(pos) for pos in liquid_positions])

num_of_gas_samples = int(np.round(Lx/gas_sample_dx))
gas_positions = np.concatenate([gas_sample_positions - np.array([(-gas_sample_dx/2-xlo)-i*gas_sample_dx, 0, 0]) for i in range(num_of_gas_samples)])
gas_velocities = np.concatenate([gas_sample_velocities for i in range(num_of_gas_samples)])
gas_mask = np.array([gas_region(pos) for pos in gas_positions])

liquid_positions = liquid_positions[liquid_mask]

gas_positions = gas_positions[gas_mask]
#pdb.set_trace()

Atoms = []
Bonds = []
Angles = []
mol = 0
nall = 0
for pos,vel in zip(liquid_positions,liquid_velocities):
    nall+=1    
    atom = [nall,0,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],'Ar']
    Atoms.append(atom)
for pos,vel in zip(gas_positions,gas_velocities):
    nall+=1    
    atom = [nall,0,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],'Ar']
    Atoms.append(atom)

Atoms = sorted(Atoms)

atom_type_dict = dict([(typ, i) for i,typ in enumerate(sorted(list(set([item[-1] for item in Atoms]))),1)])
atom_type_num = dict.fromkeys(atom_type_dict.keys(),0)

for atom in Atoms:
    atom_type_num[atom[-1]]+=1

print('\nAtoms:')
for key,value in atom_type_num.items():
    print(atom_type_dict[key],key,value)

# graph = lil_matrix((nall+1,nall+1))
# for bond in Bonds:
#     id1,id2,bt = bond
#     graph[id1,id2] = graph[id2,id1] = 1

bond_type_dict = dict([(typ, i) for i,typ in enumerate(sorted(list(set([item[-1] for item in Bonds]))),1)])
bond_type_num = dict.fromkeys(bond_type_dict.keys(),0)

for bond in Bonds:
    bond_type_num[bond[-1]]+=1
print('\nBonds:')
for key,value in bond_type_num.items():
    print(bond_type_dict[key],key,value)

angle_type_dict = dict([(angle_type,i) for i,angle_type in enumerate(list(set([ang[-1] for ang in Angles])),1)])

angle_type_num = dict.fromkeys(angle_type_dict.keys(),0)

for angle in Angles:
    angle_type_num[angle[-1]]+=1
print('\nAngles:')
for key,value in angle_type_num.items():
    print(angle_type_dict[key],key,value)
q_dict = dict([('Au',0),
               ('Ar',0),
               ('HW',0.417),
               ('OW',-0.834)])

Atoms_data = []
Bonds_data = []
Angles_data = []
Velocities_data = []
for atom in Atoms:
    ID,mol,x,y,z,vx,vy,vz,typ = atom
    Atoms_data.append(('full',domain_map(x,y,z,0,0,0,Lx,Ly,Lz),q_dict[typ],ID,atom_type_dict[typ],mol))
    Velocities_data.append((ID,vx,vy,vz))
for ID,bond in enumerate(Bonds,1):
    atom1,atom2,typ = bond
    Bonds_data.append((ID,bond_type_dict[typ],atom1,atom2))
for ID,angle in enumerate(Angles,1):
    atom1,atom2,atom3,typ = angle
    Angles_data.append((ID,angle_type_dict[typ],atom1,atom2,atom3))


mass_dict = dict([('Au',197),
                  ('Ar',40),
                  ('HW',1.0),
                  ('OW',15.999)])

sigma_dict = dict([('Au',2.637),
                   ('Ar',3.405),
                   ('HW',0),
                   ('OW',3.188)])

# epsilon unit kJ/mol, 1cal=4.184J                                                                                                  
unitcal = 4.184
epsilon_dict = dict([('Au',42.58),
                     ('Ar',0.9938),
                     ('HW',0),
                     ('OW',0.102*unitcal)])

pair_coeff_list = []
for typ in sorted(atom_type_dict.keys()):
    pair_coeff_list.append([atom_type_dict[typ],epsilon_dict[typ]/unitcal,sigma_dict[typ]])
pair_coeff_list = sorted(pair_coeff_list)


kij_dict = dict([('OW-HW', 450.0*unitcal)])
r0_dict = dict([('OW-HW', 0.9572)])

bond_coeff_list = []
for typ in sorted(bond_type_dict.keys()):
    bond_coeff_list.append([bond_type_dict[typ],kij_dict[typ]/2/unitcal,r0_dict[typ]])    
bond_coeff_list = sorted(bond_coeff_list)

kijk_dict = dict([('HW-OW-HW',55.0*unitcal)])

theta0_dict = dict([('HW-OW-HW',104.52)])

angle_coeff_list = []
for typ in sorted(angle_type_dict.keys()):
    angle_coeff_list.append([angle_type_dict[typ],kijk_dict[typ]/2/unitcal,theta0_dict[typ]])    
angle_coeff_list = sorted(angle_coeff_list)

fdata = 'atoms/atoms.data'
fxyz = 'atoms/atoms.xyz'
print('writting to %s ...'%fdata)
Mass_data = [mass_dict[typ] for i, typ in sorted([(i,typ) for typ,i in atom_type_dict.items()])]
write_data(fdata,fxyz,Atoms=Atoms_data,Bonds=Bonds_data,Angles=Angles_data,\
           mass=Mass_data,Vels=Velocities_data,\
           moment=[],xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,\
           pair_coeff_list=pair_coeff_list,bond_coeff_list=bond_coeff_list,\
           angle_coeff_list=angle_coeff_list)

print('')
infile = 'in.run'
setvar(infile,'rc',rc)
setvar(infile,'Ar_atom_typ',atom_type_dict['Ar'])
setvar(infile,'T',T)
setvar(infile,'Lx',Lx)
setvar(infile,'Ly',Ly)
setvar(infile,'Lz',Lz)
setvar(infile,'L',L)
setvar(infile,'xlo',xlo)
setvar(infile,'xhi',xhi)

setvar(infile,'ylo',ylo)
setvar(infile,'yhi',yhi)

setvar(infile,'zlo',zlo)
setvar(infile,'zhi',zhi)
setvar(infile,'P0',P0)
setvar(infile,'R0',R)
setvar(infile,'KR',KR)
setvar(infile,'S',S)

print('')
