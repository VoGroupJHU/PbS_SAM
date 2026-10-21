######################
### Import modules ###
######################

import datetime
import hoomd
import coxeter
import numpy as np
import gsd.hoomd

######################
######################

##################
### Initialize ###
##################

# Define simulation device
cpu = hoomd.device.CPU()
 
# Create simulation object
sim = hoomd.Simulation(device=cpu, seed=777)

# Types
type_full_ref = ['core','patch']

##################
##################

#####################
### Read in patch ###
#####################

# Read in file - monomer core
file_name = 'rigid.txt'
f = open(file_name,'r')
message = f.readlines()
pts_patch = []
type_patch = []
ori_patch = []
for line in message:

  # Read file
  tmp_line = line
  x_tmp,y_tmp,z_tmp,sigma_patch = tmp_line.split()
  pts_tmp = [float(x_tmp),float(y_tmp),float(z_tmp)]  

  # Add
  pts_patch.append(pts_tmp)  
  type_patch.append('patch')
  ori_patch.append([1.0,0.0,0.0,0.0])

# Define patch
sigma_patch = float(sigma_patch)

# Close file
f.close()

#####################
#####################

########################
### Read in vertices ###
########################

# Read in file - monomer core
file_name = 'verts.txt'
f = open(file_name,'r')
message = f.readlines()
pts_verts = []
for line in message:

  # Read file
  tmp_line = line
  x_tmp,y_tmp,z_tmp,sigma_in,sigma_out,I_xx,I_yy,I_zz,v_poly,a_poly = tmp_line.split()
  pts_tmp = [float(x_tmp),float(y_tmp),float(z_tmp)]  

  # Add
  pts_verts.append(pts_tmp)  

# Use coxeter to get faces
shape_poly = coxeter.shapes.ConvexPolyhedron(pts_verts)

# Moment of inertia
moment_scale = (0.85E-2)*float(v_poly)
I_xx = moment_scale*float(I_xx)
I_yy = moment_scale*float(I_yy)
I_zz = moment_scale*float(I_zz)

# Particle size
sigma_core = float(sigma_in)
sigma_out = float(sigma_out)

# Polyhedra volume
v_poly = float(v_poly)

# Polyhedra in-plane area
a_poly = float(a_poly)

# Define box
Lbox = np.ceil(1.75*sigma_out)
Lbox_z = 5.0*Lbox

# Define top wall location
zwall_top = Lbox_z/2.0 - sigma_out
zwall_bot = 0.0

# Close file
f.close()

########################
########################

#############################
### Define initial system ###
#############################

# Define snapshot
snapshot = gsd.hoomd.Frame()
snapshot.particles.N = 1
snapshot.particles.position = [[0.0,0.0,sigma_out]]
snapshot.particles.typeid = [0]
snapshot.particles.moment_inertia = [[I_xx,I_yy,I_zz]]
snapshot.configuration.box = [Lbox, Lbox, Lbox_z, 0, 0, 0]
snapshot.particles.types = type_full_ref

# Create snapshot
sim.create_state_from_snapshot(snapshot,domain_decomposition=(3,3,1))

# Replicate
nrep = 20
sim.state.replicate(nx=nrep, ny=nrep, nz=1)

# Update volume and area
V_total = float(nrep)**2*v_poly
A_total = float(nrep)**2*a_poly

# Tartget assembly density
phi_target = 0.4

#############################
#############################

#########################
### Define rigid body ###
#########################

# Initialize rigid body
rigid = hoomd.md.constrain.Rigid()

# Define DNA top
rigid.body['core'] = {
    "constituent_types": type_patch,
    "positions": pts_patch,
    "orientations": ori_patch}

rigid.create_bodies(sim.state)

#########################
#########################

#########################
### Define potentials ###
#########################

# Temperature
kbt = 1.0

# Integration timestep
dt = 5.0E-4

# ALJ Interaction
epsilon_alj = 0.1

# NP-NP interaction
epsilon = 1.0
epsilon_att = 15.0

# NP-Wall interaction
eps_wall_rep = 1.0
eps_wall_att = 10.0

# Set up integrator
integrator = hoomd.md.Integrator(dt=dt,integrate_rotational_dof=True)
integrator.rigid = rigid
sim.operations.integrator = integrator

# Cell list
cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=['body'])

# Mixing parameter
sigma_core_patch = 0.5*(sigma_core+sigma_patch)

### ALJ Cutoffs ###
# Repulsive interaction - ALJ cutoffs
buffer_factor = 1.15
distance_factor = 2.0**(1.0/6.0) 
rcut_core_rep_alj = buffer_factor*distance_factor*sigma_core*(sigma_out/sigma_core)
rcut_core_patch_rep_alj = buffer_factor*distance_factor*(0.5*sigma_core*(sigma_out/sigma_core)+0.5*sigma_patch)
rcut_patch_rep_alj = buffer_factor*distance_factor*sigma_patch
# Attractive interaction - ALJ cutoffs
distance_factor = 3.0
rcut_core_att_alj = distance_factor*(sigma_core*(sigma_out/sigma_core))
rcut_core_patch_att_alj = buffer_factor*distance_factor*(0.5*sigma_core*(sigma_out/sigma_core)+0.5*sigma_patch)
rcut_patch_att_alj = buffer_factor*distance_factor*sigma_patch
###################

### Mie Cutoffs ###
# Mie potential parameter
mie_n = 12.0
mie_m = float(8)
# Cutoff
cutoff_rep_mie = (mie_n/mie_m)**(1.0/(mie_n-mie_m))
cutoff_att_mie = 3.0*((mie_n/mie_m)**(1.0/(mie_n-mie_m)))/(2.0**(1.0/6.0))
# Repulsive interaction - Mie cutoffs
rcut_core_rep_mie = cutoff_rep_mie*sigma_core
rcut_patch_rep_mie = cutoff_rep_mie*sigma_patch
rcut_core_patch_rep_mie = cutoff_rep_mie*sigma_core_patch
# Attractive interaction - Mie cutoffs
rcut_core_att_mie = cutoff_att_mie*sigma_core
rcut_patch_att_mie = cutoff_att_mie*sigma_patch
rcut_core_patch_att_mie = cutoff_att_mie*sigma_core_patch
###################

### Mie Cutoffs - Wall ###
# Mie potential parameter
mie_n = 12.0
mie_m_wall = float(6)
# Cutoff
cutoff_rep_mie_wall = (mie_n/mie_m_wall)**(1.0/(mie_n-mie_m_wall))
cutoff_att_mie_wall = 3.0*((mie_n/mie_m_wall)**(1.0/(mie_n-mie_m_wall)))/(2.0**(1.0/6.0))
# Repulsive interaction - Mie cutoffs
rcut_core_rep_mie_wall = cutoff_rep_mie_wall*sigma_core
rcut_patch_rep_mie_wall = cutoff_rep_mie_wall*sigma_patch
rcut_core_patch_rep_mie_wall = cutoff_rep_mie_wall*sigma_core_patch
# Attractive interaction - Mie cutoffs
rcut_core_att_mie_wall = cutoff_att_mie_wall*sigma_core
rcut_patch_att_mie_wall = cutoff_att_mie_wall*sigma_patch
rcut_core_patch_att_mie_wall = cutoff_att_mie_wall*sigma_core_patch
##########################

### Define ALJ potential ###
mie_factor_sphere = 1.05;
# Initialize
alj = hoomd.md.pair.aniso.ALJ(nlist=cell,default_r_cut=rcut_core_rep_alj)
# Vertices
alj.shape['core'] = dict(vertices=shape_poly.vertices, faces=shape_poly.faces, rounding_radii=[0.0,0.0,0.0])
alj.shape['patch'] = dict(vertices=[], faces=[], rounding_radii=[sigma_patch/2.0,sigma_patch/2.0,sigma_patch/2.0])
# Pairwise interaction
alj.params[('core','core')] = dict(epsilon=epsilon_alj, sigma_i=mie_factor_sphere*sigma_core, sigma_j=sigma_core, alpha=0)
alj.params[('patch','patch')] = dict(epsilon=epsilon_alj, sigma_i=sigma_patch, sigma_j=sigma_patch, alpha=0)
alj.params[('core','patch')] = dict(epsilon=epsilon_alj, sigma_i=mie_factor_sphere*sigma_core, sigma_j=sigma_patch, alpha=0)
# Cutoffs
alj.r_cut[('core','core')] = mie_factor_sphere*rcut_core_rep_alj
alj.r_cut[('patch','patch')] = rcut_patch_rep_alj
alj.r_cut[('core','patch')] = mie_factor_sphere*rcut_core_patch_rep_alj
# Add
integrator.forces.append(alj)
############################

### Define Mie potential ###
# Initialize
mie = hoomd.md.pair.Mie(nlist=cell,default_r_cut=mie_factor_sphere*rcut_core_rep_mie)
# Pairwise interaction
mie.params[('core','core')] = dict(epsilon=epsilon, sigma=mie_factor_sphere*sigma_core, n=mie_n, m=mie_m)
mie.params[('patch','patch')] = dict(epsilon=epsilon, sigma=sigma_patch, n=mie_n, m=mie_m)
mie.params[('core','patch')] = dict(epsilon=epsilon, sigma=mie_factor_sphere*sigma_core, n=mie_n, m=mie_m)
# Cutoffs
mie.r_cut[('core','core')] = 0.001
mie.r_cut[('patch','patch')] = 0.001
mie.r_cut[('core','patch')] = 0.001
# Add
integrator.forces.append(mie)
#################################

### Define wall ###
# Top wall
wall_top = hoomd.wall.Plane(origin=(0.0,0.0,zwall_top), normal=(0.0,0.0,-1.0))
# Bottom wall
wall_bot = hoomd.wall.Plane(origin=(0.0,0.0,zwall_bot), normal=(0.0,0.0,1.0))
# Define wall
lj_wall_top = hoomd.md.external.wall.Mie(walls=[wall_top])
lj_wall_bot = hoomd.md.external.wall.Mie(walls=[wall_bot])
# Interaction - top
lj_wall_top.params['core'] = dict(epsilon=eps_wall_rep, sigma=sigma_core, n=mie_n, m=mie_m, r_cut=rcut_core_rep_mie)
lj_wall_top.params['patch'] = dict(epsilon=eps_wall_rep, sigma=sigma_patch, n=mie_n, m=mie_m, r_cut=rcut_patch_rep_mie)
integrator.forces.append(lj_wall_top)
# Interaction - bottom
lj_wall_bot.params['core'] = dict(epsilon=eps_wall_att, sigma=sigma_core, n=mie_n, m=mie_m_wall, r_cut=rcut_core_att_mie_wall)
lj_wall_bot.params['patch'] = dict(epsilon=eps_wall_att, sigma=sigma_patch, n=mie_n, m=mie_m_wall, r_cut=rcut_patch_att_mie_wall)
integrator.forces.append(lj_wall_bot)
###################

### Define ensemble and add to integrator ###
# Integration set
full = hoomd.filter.All()
group_integrate = hoomd.filter.Type(['core'])
# Integrator
nvt = hoomd.md.methods.ConstantVolume( filter=group_integrate, thermostat=hoomd.md.methods.thermostats.MTTK(kT=kbt, tau=0.1) )
# Add integrator
integrator.methods.append(nvt)
# Thermalize
sim.state.thermalize_particle_momenta(filter=group_integrate, kT=kbt)
#############################################

### Write to file ###
# Shape logging
logger_shape = hoomd.logging.Logger()
logger_shape.add(alj,quantities=['type_shapes'])
# GSD dump
ndump = int(1E4)
gsd_writer = hoomd.write.GSD(filename='sample.gsd',
                             trigger=hoomd.trigger.Periodic(ndump),
                             mode='wb',
                             logger=logger_shape)
sim.operations.writers.append(gsd_writer)
#####################

### Define logger for tps printing ###
logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps'])
class Status():
    def __init__(self, sim):
        self.sim = sim
    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0
    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))
status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=5000),
                          logger=logger)
sim.operations.writers.append(table)
######################################

#########################
#########################

###########
### Run ###
###########

# Prerun
sim.run(0,write_at_start=True)

### Equilibration ###
print('Equilibrate')
nequil = 3E5
num_loop = int(float(nequil)/float(ndump))
for i in range(0,num_loop): 
    gsd_writer.flush()
    sim.run(ndump)  
#####################

### Downward force due to wall ###
print('Gravity')
# Define force
constant = hoomd.md.force.Constant(filter=full)
# Define vector
gfactor = -0.5
constant.constant_force['A'] = (0.0,0.0,gfactor)
# Add wall driving force
integrator.forces.append(constant)
# Run downward force
nequil = 3E5
num_loop = int(float(nequil)/float(ndump))
for i in range(0,num_loop): 
    gsd_writer.flush()
    sim.run(ndump) 
##################################

### Compress ###
print('Compress')
# Define ramp
nramp = int(2E5)
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=nramp)
# Define box updater
initial_box = sim.state.box
# Define final box
final_box = hoomd.Box.from_box(initial_box)
area_final = A_total/phi_target
Lxy_final = np.sqrt(area_final)
final_box.Lx = Lxy_final
final_box.Ly = Lxy_final
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
# Run compression
nequil = 2.0*float(nramp)
num_loop = int(float(nequil)/float(ndump))
for i in range(0,num_loop): 
    gsd_writer.flush()
    sim.run(ndump)  
################

### Update attraction ###
# Thermalize
sim.state.thermalize_particle_momenta(filter=group_integrate, kT=kbt)
print('Attraction')
# mie.params[('core','core')] = dict(epsilon=epsilon_att, sigma=mie_factor_sphere*sigma_core, n=mie_n, m=mie_m)
mie.params[('patch','patch')] = dict(epsilon=epsilon_att, sigma=sigma_patch, n=mie_n, m=mie_m)
# mie.params[('core','patch')] = dict(epsilon=epsilon_att, sigma=mie_factor_sphere*sigma_core, n=mie_n, m=mie_m)
# mie.r_cut[('core','core')] = rcut_core_att_mie
mie.r_cut[('patch','patch')] = rcut_patch_att_mie
# mie.r_cut[('core','patch')] = rcut_core_patch_att_mie
# Run
nequil = 1E5
num_loop = int(float(nequil)/float(ndump))
for i in range(0,num_loop): 
    gsd_writer.flush()
    sim.run(ndump)
#########################

### Cycle ###
print('Thermal Cycle')
# Clear previous NVT
integrator.methods.clear()
# High temperature
kbt_low = 1.0*kbt
kbt_high = 2.00
# Hold and ramp time
time_hold = 1.5
time_ramp = 0.5
n_hold = int(time_hold*1E5)
n_ramp = int(time_ramp*1E5)
# Define cycle
ramp_kT = hoomd.variant.Cycle(A=kbt_low,B=kbt_high,t_start=sim.timestep,t_A=n_hold,t_AB=n_ramp,t_B=n_hold,t_BA=n_ramp)
# Clear previous NVT
integrator.methods.clear()
# Update integrator
nvt = hoomd.md.methods.ConstantVolume( filter=group_integrate, thermostat=hoomd.md.methods.thermostats.MTTK(kT=ramp_kT, tau=0.1) )
# Add integrator
integrator.methods.append(nvt)
# Run cycle
nrun = 10E7
ncycle = int(float(2.0*time_hold*1E5+2.0*time_ramp*1E5)/float(ndump))
num_loop = int(float(nrun)/float(ndump))
for i in range(0,num_loop):
    if i % ncycle == 0:
        print('New Cycle')
    gsd_writer.flush()
    sim.run(ndump)
############

###########
###########

