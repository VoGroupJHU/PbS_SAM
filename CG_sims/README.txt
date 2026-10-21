README for CG Simulation

This folder contains an example simulation script to perform the CG NC assembly simulations shown in the main text

Descriptions of files are:

%%%%%%%%%
TXT Files
%%%%%%%%%

Initial configuration files to perform simulation

verts.txt
	File describing the vertices of the CG NC shape

rigid.txt
	File describing surface points to define CG potential and allow for quick orientation calculation for post-processing

%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%
%%% PYTHON Script %%%
%%%%%%%%%%%%%%%%%%%%%

Simulations are performed using the HOOMD-Blue Package v4.9.1

sims_run.py
	Simulation script that creates an initial configuration of the coarse-grained NCs, compresses the system, and perform the assembly simulation run.

orientation_analyzer.py
	Analysis script for computing orientation and RDF of the self-assembled NCs.

%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulation Output %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

Example GSD output files for spherical SNP

sample_data.gsd
	GSD file (binary type file) containing position and orientations of all particles ran in simulation. This file serves as input to analysis script provided.

	Reading GSD files are readily available using the OVITO visualization software or using the GSD Python package

%%%%%%%%%%%%%%%%%%%%%%%%%