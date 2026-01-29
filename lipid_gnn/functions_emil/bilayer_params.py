'''
The following code are the default bilayer params.
Please do not change the name of the functions and variables.
'''

###############################################################################
# Bilayer params ##############################################################
###############################################################################
import glob

systems_path = f'systems/full_elastic_network'
pdb = 'structures/ire1_martini_2monomer_sep_elastic.pdb'
membrane_comp = {'POPC': 100}
new_protein_label = {'PROA': 1,
                     'PROB': 1}
x = 15
y = 15
z = 13
salt = 0.15
# gmx_index = 'gmx21_mpi make_ndx -nobackup'
# gmx_mdrun = 'gmx21_mpi mdrun -pin on -ntomp 10 -dlb yes -nobackup -v'
# gmx_grompp = 'gmx21_mpi grompp -maxwarn 5 -nobackup' 
gmx_index = 'gmx21 -nobackup make_ndx'
gmx_mdrun = 'gmx21 -nobackup mdrun -pin on '
gmx_grompp = 'gmx21 -nobackup grompp -maxwarn 5' 
stateA = 0.4
stateB = 5.0
check_every = 60*2
drmsd_reference = 'ire1_dimer_drmsd_ref/connections_cut_ref.npy'
toppar_path = 'sim_params/toppar'
min_setup = sorted(glob.glob('sim_params/minimization_setup/*'))
equi_setup = sorted(glob.glob('sim_params/equibrilation_setup/*'))
mdp_path = 'sim_params/run.mdp'
aimmd_params_path = 'aimmd_params'
def stop(subtrajectory:str, stateA:float, drmsd_reference:str=drmsd_reference):
    global frame, index, traj_con
    while True:
        try:
            frame = md.load_frame(subtrajectory, index=index, top=frame)
            drmsd = DRMSD(frame, drmsd_reference)
            if drmsd < stateA:
                return True
            traj_con = np.append(traj_con, np.array([[int(index), float(drmsd)]]), axis=0)
            index += 1
        except:
            return False
