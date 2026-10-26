from squid.lammps import job
import os
from squid.forcefields.parameters import Parameters


def submit_job(run_name, paramsfile):
    '''
    Function for calling SLURM job scheduler to run LAMMPS input
    '''
    # Specify job settings
    datafile = run_name + ".data"
    queue = "parallel"
    nodes = 3
    ntasks = 144
    walltime = "72:00:00"
    dpath = os.getcwd() + "/"

    ###########################################################
    # For SMRFF parameters

    # Define parameters and data file path
    params = Parameters(
        [
            "xPb", "xS", "OS", "C", "OC", "CH2", "CH", "CH3", "N", "C1", "C2",
            "C3", "P", "O1", "O2", "C4", "C5", "O3", "C6", "O4", "C7", "C8",
            "C9", "C10"
        ],
        smrff_file=paramsfile,
        adjust_range=True
    )
    params.set_smoothed_pair_potentials([
        ("morse", 6.5, "sin_r"),
        ("lj/cut/coul/long", 12.0, "sin_l")
    ])

    ###########################################################
    script = '''
    # MAIN SETTINGS
    clear
    units real
    atom_style full
    bond_style harmonic
    angle_style harmonic
    dihedral_style opls
    improper_style harmonic
    kspace_style pppm 1.0e-4
    kspace_modify slab 3.0
    boundary p p f
    dielectric 1.88

    # READ DATA FILE AND DEFINE PARAMETERS
    read_data ''' + dpath + datafile + '''
    pair_style smrff morse 6.5 lj/cut/coul/long 12.0 smooth sin_r sin_l
    pair_modify mix arithmetic
    ''' + params.dump_style('all', tstyle_smrff=True, write_file=True) + '''

    # FIXES FOR GROUPS OF ATOMS
    group heads type 9
    group other subtract all heads
    group nc1 id 1:6848
    group nc2 id 6849:13696
    group mono type 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    fix z_walls all wall/reflect zlo EDGE zhi EDGE

    # Neighbor settings and minimization
    neighbor 12.0 bin
    neigh_modify delay 0 every 1 check yes one 5000
    min_style sd
    minimize 1e-4 1e-6 100 1000
    min_style cg 
    min_modify line quadratic
    minimize 1e-5 1e-7 1000 10000

    # Calculations for PMF
    compute 1 nc1 group/group mono kspace yes
    compute 2 nc2 group/group mono kspace yes
    compute 3 nc1 group/group nc2 kspace yes

    # Calculations for distance
    compute 4 nc1 com
    compute 5 nc2 com
    compute 6 mono com
    # NC1 coordinates
    variable nc1_x equal c_4[1]
    variable nc1_y equal c_4[2]
    variable nc1_z equal c_4[3]
    # NC2 coordinates
    variable nc2_x equal c_5[1]
    variable nc2_y equal c_5[2]
    variable nc2_z equal c_5[3]
    # Monolayer coordinates
    variable mono_x equal c_6[1]
    variable mono_y equal c_6[2]
    variable mono_z equal c_6[3]
    # Inter-NC distance
    variable dx_nc equal v_nc1_x-v_nc2_x
    variable dy_nc equal v_nc1_y-v_nc2_y
    variable dz_nc equal v_nc1_z-v_nc2_z
    variable dist_nc equal sqrt(v_dx_nc*v_dx_nc+v_dy_nc*v_dy_nc+v_dz_nc*v_dz_nc)
    # Vertical separations (Z-direction)
    variable sep_nc1_mono equal v_nc1_z-v_mono_z
    variable sep_nc2_mono equal v_nc2_z-v_mono_z
    # Average separation (arithmetic mean of both NCs)
    variable avg_sep_nc_mono equal 0.5*(v_sep_nc1_mono+v_sep_nc2_mono)
    
    # Thermodynamic output
    thermo 1000
    thermo_style custom step time temp press etotal c_1 c_2 c_3 v_dist_nc v_avg_sep_nc_mono

    # Stage 1 - Run at 100K for 1 ps
    timestep 0.5
    fix freeze_heads heads setforce NULL NULL 0.0
    velocity heads set 0.0 0.0 0.0
    velocity other create 100.0 12345 rot yes dist gaussian
    fix 1 other langevin 100.0 100.0 100.0 12345
    fix 2 other nve/limit 0.05
    run 2000
    unfix 1
    unfix 2

    # Stage 2 - warm up to 423K for 1 ps
    fix 1 other langevin 100.0 423.15 100.0 23456
    fix 2 other nvt temp 100.0 423.15 100.0
    run 2000
    unfix 1
    unfix 2

    # Production: NVT at 423 K
    dump 2 all xyz 10000 dump.xyz
    dump_modify 2 element Pb S O C O C C C N C C C P O O C C O C O C C C C
    dump_modify 2 sort id
    restart 10000 restart.*.equil
    reset_timestep 0
    timestep 2.0
    fix 1 other langevin 423.15 423.15 100.0 34567
    fix 2 other nvt temp 423.15 423.15 100.0
    run 50000000
    '''

    job(run_name,
        script,
        queue=queue,
        nodes=nodes,
        ntasks=ntasks,
        walltime=walltime,
        pair_coeffs_in_data_file=False)


if __name__ == '__main__':
    # Specify input files to submit
    list_of_files = [
        "n327_dppc_1.data"
    ]
    for i in list_of_files:
        if ".data" in i:
            job_name = i.split('.')[0]
            submit_job(job_name, "params.smrff")