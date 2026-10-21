###############
### Modules ###
###############

import os
import numpy as np
import random
import freud
import gsd.pygsd
import gsd.hoomd 
import matplotlib.pyplot as plt
from scipy import stats

###############
###############

###############################
### Define phase space grid ###
###############################

current_path = os.getcwd()

# Interaction range (n exponent)
n_array = [2,3,4,5,6,7,8,9,10,11]

# Interaction range (n exponent)
nwall_array = [2,3,4,5,6,7,8,9,10,11]

# Add extra
flag_add_extra = 0

# # Subselect
# n_array = [3]
# nwall_array = [11]

# Define type
type_core = 0
type_patch = 1

# Define phase space
phase_space = []
for n in n_array:
	for nwall in nwall_array:
		tmp = [float(n),float(nwall)]
		phase_space.append(tmp)

###############################
###############################

##########################
### Define rdf compute ###
##########################

def compute_rdf(box_arr, points_arr, prop, r_max=15.0, bins=200, label=None, ax=None):
    """Helper function for plotting RDFs."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_title(prop, fontsize=16)
    rdf = freud.density.RDF(bins, r_max)
    for box, points in zip(box_arr, points_arr):
        rdf.compute(system=(box, points), reset=False)
    return rdf

def dr_periodic(box_size,pts1,pts2):

	# X Direction
	dr_x = pts2[0] - pts1[0]
	if dr_x	> 0.5*box_size[0]:
		dr_x = dr_x - box_size[0]
	
	# Y Direction
	dr_y = pts2[1] - pts1[1]
	if dr_y	> 0.5*box_size[1]:
		dr_y = dr_y - box_size[1]
	
	# Z Direction
	dr_z = pts2[2] - pts1[2]
	if dr_z	> 0.5*box_size[2]:
		dr_z = dr_z - box_size[2]

	# Output
	dr_final = np.array([dr_x,dr_y,dr_z])
	return dr_final

##########################
##########################

################################
### Load in data and analyze ###
################################

for pp in range(0,len(phase_space)):

	### Define path ###	
	# Grab parameter
	tmp = phase_space[pp]
	# Convert parameter	
	n_str = '{:.0f}'.format(tmp[0])
	nwall_str = '{:.0f}'.format(tmp[1])
	# Define data folder
	data_path = current_path + '/n' + str(n_str) + '/nwall' + str(nwall_str)
	file_name = data_path + '/sample.gsd'
	verts_name = data_path + '/verts.txt'
	print(data_path)
	###################

	### Read in vertices ###	
	# Read in file - monomer core
	f = open('verts.txt','r')
	message = f.readlines()
	for line in message:
		# Read file
		tmp_line = line
		x_tmp,y_tmp,z_tmp,sigma_in,sigma_out,I_xx,I_yy,I_zz,v_poly,a_poly = tmp_line.split()
	# Convert
	sigma_out = float(sigma_out)
	sigma_in = float(sigma_in)
	# Close file
	f.close()
	########################

	### Read in patch ###
	f = open('rigid.txt','r')
	message = f.readlines()
	for line in message:
	  # Read file
	  tmp_line = line
	  x_tmp,y_tmp,z_tmp,sigma_patch = tmp_line.split()
	# Define patch
	sigma_patch = float(sigma_patch)
	# Close file
	f.close()
	#####################

	# Read in data
	ff = gsd.pygsd.GSDFile(open(file=file_name,mode='rb'))
	t = gsd.hoomd.HOOMDTrajectory(ff);
	
	### Cycling and frame definition ###
	# Dump rate
	ndump = float(1E4)
	# Initial equilibration
	ninit = 3E5 + 3E5
	# Compression run
	ncompress = 2.0*2E5
	# Post compression equilibriation
	ncompress_post = 1E5
	# Cycle time
	nstep = float(1E5)
	time_hold = 1.5
	time_ramp = 0.5	
	ncycle = float(2.0*time_hold*nstep+2.0*time_ramp*nstep)
	# Determine start time of cycling
	nstart_cycle = ninit + ncompress + ncompress_post
	#####################################

	### Define sampling frequency ###	
	# Start sampling in middle of low temperature range
	nsampling_start = nstart_cycle + 0.5*nstep
	# Convert to frame (shift by 1 b/c of zero index)
	nsampling_start_frame = nsampling_start / ndump - 1.0
	# Cycle time in terms of frame 
	ncycle_frame = ncycle / ndump
	# Total frame
	ntotal = float(len(t))
	# Create sampling array
	nsampling_frame = np.arange(nsampling_start_frame,ntotal,ncycle_frame)
	if flag_add_extra == 1:
		nsampling_frame2 = nsampling_frame + 1
		nsampling_frame3 = nsampling_frame - 1
		nsampling_frame5 = nsampling_frame - 2
		combined_set = 	nsampling_frame.tolist() + nsampling_frame2.tolist() + nsampling_frame3.tolist() + nsampling_frame5.tolist()
		nsampling_frame = list(set(combined_set))
	nsampling_frame.sort()
	#################################

	#########################################################
	### Compute particle orientation/alignment properties ###
	#########################################################

	# Average over only later portion of simulation
	index_start = np.floor(0.2*len(nsampling_frame))
	index_start = int(index_start)	
	# Initialize
	angle_store = []
	angle_zdir_store = []
	angle_tilt_store = []
	# Loop through frames	
	for i in range(index_start,len(nsampling_frame)):

		# Logging
		print(pp,len(phase_space),i,int(nsampling_frame[i]),len(nsampling_frame))

		### Get frame specific particle parameters ###		
		# Get frame counter
		t_tmp = int(nsampling_frame[i])
		# Grab frame
		frame_tmp = t[t_tmp]
		# Grab box
		box_tmp = frame_tmp.configuration.box		
		# Grab position
		pts_tmp = frame_tmp.particles.position
		type_tmp = frame_tmp.particles.typeid
		body_tmp = frame_tmp.particles.body	
		##############################################

		### Construct compute features ###		
		# Construct 2D Box
		box_2D_tmp = freud.box.Box(box_tmp[0], box_tmp[1], Lz=0, is2D=True)
		box_3D_tmp = freud.box.Box(box_tmp[0], box_tmp[1], box_tmp[2])	
		##################################

		### Grab relevant data ###
		# Get patch index
		indx_patch_tmp = np.where(type_tmp == type_patch)[0]
		# Get core index
		indx_core_tmp = np.where(type_tmp == type_core)[0]
		# Define positions
		pts_patch_tmp = pts_tmp[indx_patch_tmp,:]
		pts_core_tmp = pts_tmp[indx_core_tmp,:]
		# Define body
		body_patch_tmp = body_tmp[indx_patch_tmp]
		body_core_tmp = body_tmp[indx_core_tmp]
		##########################

		### Determine alignment of clusters ###
		# Check direction
		zdir_check = [0.0,0.0,1.0]
		# Cluster cores for connect particles
		system_core = freud.AABBQuery(box_3D_tmp, pts_core_tmp)
		cl_core_tmp = freud.cluster.Cluster()
		cl_core_tmp.compute(system_core, neighbors={"r_max": 1.375*sigma_in})
		# Get number of clusters
		N_core_cluster_tmp = cl_core_tmp.num_clusters
		# Cluster index
		indx_core_cluster_tmp = cl_core_tmp.cluster_idx
		# loop through cluster		
		for nn in range(0,N_core_cluster_tmp):
			# Grab points
			indx_tmp = np.where(indx_core_cluster_tmp == nn)[0]
			# Check pairwise binding
			if len(indx_tmp) > 1:
				# Grab relevant particles
				body_core_cluster_tmp = body_core_tmp[indx_tmp]
				pts_core_cluster_tmp = pts_core_tmp[indx_tmp,:]
				# Pre-averaging
				angle_zdir_avg_tmp = 0.0
				# Loop through cores
				for ii in range(0,len(body_core_cluster_tmp)):
					# Get patches
					indx_patch_tmp = np.where(body_patch_tmp == body_core_cluster_tmp[ii])[0]
					# Get all patches
					pts_patch_cluster_tilt_tmp = pts_patch_tmp[indx_patch_tmp,:]
					# Compute displacement
					dr_core_patch_tmp = pts_patch_cluster_tilt_tmp - pts_core_cluster_tmp[ii,:]
					# Find max
					indx_zdir_tmp = np.argmax(dr_core_patch_tmp[:,2])
					# Max displacement vector
					dr_zdir_tmp = dr_core_patch_tmp[indx_zdir_tmp,:]
					# Convert to angle
					angle_zdir_tmp = np.acos(np.dot(dr_zdir_tmp,zdir_check)/np.linalg.norm(zdir_check)/np.linalg.norm(dr_zdir_tmp))
					angle_zdir_tmp = angle_zdir_tmp*(180.0/np.pi)
					# Average
					angle_zdir_avg_tmp = angle_zdir_avg_tmp + angle_zdir_tmp/float(len(body_core_cluster_tmp))
				# Store
				angle_tilt_store.append([float(angle_zdir_avg_tmp),float(len(body_core_cluster_tmp))])
		#######################################

		### Determine misalignment ###				
		# Cluster patches for connect particles
		system_patch = freud.AABBQuery(box_3D_tmp, pts_patch_tmp)
		cl_patch_tmp = freud.cluster.Cluster()
		cl_patch_tmp.compute(system_patch, neighbors={"r_max": 1.2*sigma_patch})
		# Get number of clusters
		N_patch_cluster_tmp = cl_patch_tmp.num_clusters
		# Cluster index
		indx_patch_cluster_tmp = cl_patch_tmp.cluster_idx
		# Reset body log
		core_logger = []		
		for nn in range(0,N_patch_cluster_tmp):
			# Grab points
			indx_tmp = np.where(indx_patch_cluster_tmp == nn)[0]
			# Check pairwise binding
			if len(indx_tmp) > 1:

				# Reset position - XY plane
				body_core_cluster_tmp = []
				pts_core_cluster_tmp = []				
				pts_patch_cluster_tmp = []

				# Reset position - Z plane
				pts_core_cluster_zdir_tmp = []
				pts_patch_cluster_zdir_tmp = []

				# Get body for each patch
				body_cluster_patch_tmp = body_patch_tmp[indx_tmp]

				### Get misalignment in Z plane ###
				zdir_check = [0.0,0.0,1.0]
				for ii in range(0,len(body_cluster_patch_tmp)):
					if body_cluster_patch_tmp[ii] not in core_logger:
						# Update logger
						core_logger.append(int(body_cluster_patch_tmp[ii]))
						# Get body tag
						indx_core_tmp = np.where(body_core_tmp == body_cluster_patch_tmp[ii])[0]
						indx_patch_tmp = np.where(body_patch_tmp == body_cluster_patch_tmp[ii])[0]
						# Get core
						pts_core_cluster_zdir_tmp.append(pts_core_tmp[indx_core_tmp,:])
						# Get all patches
						pts_patch_cluster_zdir_tmp.append(pts_patch_tmp[indx_patch_tmp,:])
						# Compute displacement
						dr_core_patch_tmp = pts_patch_cluster_zdir_tmp[0] - pts_core_cluster_zdir_tmp[0]
						# Find max
						indx_zdir_tmp = np.argmax(dr_core_patch_tmp[:,2])
						# Max displacement vector
						dr_zdir_tmp = dr_core_patch_tmp[indx_zdir_tmp,:]
						# Convert to angle
						angle_zdir_tmp = np.acos(np.dot(dr_zdir_tmp,zdir_check)/np.linalg.norm(zdir_check)/np.linalg.norm(dr_zdir_tmp))
						angle_zdir_tmp = angle_zdir_tmp*(180.0/np.pi)						
						# Store
						angle_zdir_store.append(angle_zdir_tmp)
				###################################

				### Get misalignment in XY plane ###				
				# Get body tags and corresponding core position
				for ii in range(0,len(body_cluster_patch_tmp)):
					# Get body tag
					indx_core_tmp = np.where(body_core_tmp == body_cluster_patch_tmp[ii])[0]					
					# Get core and patch XYZ
					pts_core_cluster_tmp.append(pts_core_tmp[indx_core_tmp,:])
					pts_patch_cluster_tmp.append(pts_patch_tmp[indx_tmp[ii],:])
				# Define patch position
				pts_patch_cluster_tmp = pts_patch_tmp[indx_tmp,:]
				# Define core position
				pts_core_cluster_tmp = np.array(pts_core_cluster_tmp)
				# Compute directional vector
				dr1_tmp = pts_patch_cluster_tmp[0,:] - pts_core_cluster_tmp[0,:]
				dr2_tmp = pts_patch_cluster_tmp[1,:] - pts_core_cluster_tmp[1,:]
				# Grab first element
				dr1_tmp = dr1_tmp[0]
				dr2_tmp = dr2_tmp[0]
				# Get displacement angle
				angle_tmp = np.acos(np.dot(dr1_tmp,dr2_tmp)/np.linalg.norm(dr1_tmp)/np.linalg.norm(dr2_tmp))
				angle_tmp = angle_tmp*(180.0/np.pi)
				# Convert to displacement from perfect
				angle_tmp = np.min( [abs(180.0-angle_tmp), angle_tmp] )
				# Store
				angle_store.append(float(angle_tmp))
				####################################
		##############################
	
	### Final monolayer tilt angle averaging ###
	# Convert to array
	angle_tilt_store = np.array(angle_tilt_store)
	# Define weights and averaging
	w_tilt_avg = angle_tilt_store[:,1]/np.sum(angle_tilt_store[:,1])
	angle_tilt_avg = float(np.sum(angle_tilt_store[:,0]*w_tilt_avg))	
	############################################		

	### Write to file ###	
	# Go to folder
	os.chdir(data_path)
	### Write misalignment angle - XY ###	
	# Misalignment angle - XY direction
	rdf_name = 'angle_xy.txt'
	fo = open(rdf_name,"w+")
	for i in range(0,len(angle_store)):
		fo.write( "%e\n" % ( angle_store[i] ) )
	# Close order
	fo.close()
	#####################################
	### Write misalignment angle - Z ###		
	# Misalignment angle - Z direction
	rdf_name = 'angle_z.txt'
	fo = open(rdf_name,"w+")
	for i in range(0,len(angle_zdir_store)):
		fo.write( "%e\n" % ( angle_zdir_store[i] ) )
	# Close order
	fo.close()
	####################################
	### Write layer tilt angle ###		
	# Layer tilt angle
	rdf_name = 'angle_tilt.txt'
	fo = open(rdf_name,"w+")
	for i in range(0,len(angle_tilt_store)):
		fo.write( "%e %e\n" % ( angle_tilt_store[i,0], angle_tilt_store[i,1] ) )
	# Close order
	fo.close()
	##############################
	# Go back
	os.chdir(current_path)
	#####################

	################################
	### End property calculation ###
	################################


################################
################################

