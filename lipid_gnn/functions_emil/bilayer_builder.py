import os
import sys
from bilayer_functions import *


def build_bilayer(params_path:str='bilayer_params.py', new_params:dict={}):
    
    
    """
    Builds a membrane bilayer system and prepares it for molecular dynamics simulations, including minimization,
    equilibration, initial path generation, and AIMMD setup.

    Parameters
    ----------
    params_path : str, optional
        Path to the Python file containing default bilayer parameters. Default is 'bilayer_params.py'.
    new_params : dict, optional
        Dictionary of parameters to override the default values.

    Returns
    -------
    None
        This function executes system setup and simulation preparation steps but does not return any object.
        All generated files, directories, and logs are stored in the appropriate system-specific folders.

    Notes
    -----
    - The function performs the following steps:
      1. Loads default bilayer parameters and overrides with `new_params`.
      2. Sets up directories for the system, equilibration, minimization, and initial paths.
      3. Uses insane.py to build the bilayer.
      4. Corrects the topology file with updated protein labels.
      5. Prepares an index file for GROMACS simulations.
      6. Performs energy minimization steps using GROMACS.
      7. Performs equilibration steps using GROMACS.
      8. Generates initial transition paths with the provided CV range.
      9. Prepares an AIMMD folder and copies all relevant files for further path ensemble analysis.
    - Requires external programs such as GROMACS and insane.py to be installed and available in the system PATH.
    - Relies on several global variables and files (e.g., `membrane_comp`, `pdb`, `x`, `y`, `z`, `toppar_path`, `stateA`, `stateB`, `check_every`) to be defined in the parameter file or `new_params`.
    - The function is intended to automate setup and execution for membrane-related molecular dynamics workflows.
    """

    ###############################
    # Get correct params ##########
    ###############################
    
    # get default params
    exec(open(params_path, 'r').read(), globals())
    # get new params
    if type(new_params) == dict:
        for value in new_params:
            exec(f'{value} = {new_params[value]}', globals())
    else:
        print(f'new_params has to be type dict, you provided {type(new_params)}')
    
    
    ###############################
    # Set up directories ##########
    ###############################
    
    # make new directory
    systems = mkdir(systems_path)
    name  = '_'.join([f'{key}{membrane_comp[key]}'for key in membrane_comp])
    mcomp = ' '.join([f'-l {key}:{membrane_comp[key]}'for key in membrane_comp])
    
    # making input log file
    input_log = {'pdb':pdb,
                 'membrane composition':name,
                 'xyz':f'{x} {y} {z}',
                 'salt': salt,
                 'toppar_path': toppar_path}
    
    # make new directory
    di = mkdir(f'{systems}/{name}')
    equi_dir = mkdir(f'{di}/equilibration')
    min_dir = mkdir(f'{di}/minimization')
    initial_dir = mkdir(f'{di}/initial')
    os.system(f'rm -rf {di}/toppar')
    os.system(f'cp -r {toppar_path} {di}/toppar')
     
    ###############################
    # Build bilayer ###############
    ###############################
    
    # Do insane.py
    insane_input = (f'python2 functions/insane.py -f {pdb} -o {di}/run.gro '
                    f'-x {x} -y {y} -z {z} {mcomp} -center -sol W '
                    f'-salt {salt} -charge auto -p {di}/topol.top')

    print(trun(insane_input, verbose=False))
    input_log['insane input'] = insane_input 
    save_log(di, input_log)

    ###############################
    # correct topology ############
    ###############################
    
    # get lines from the top file
    top = [line.strip('\n') for line in open(f'{di}/topol.top', 'r').readlines()
          if line[:8] != '#include']

    # get new protein label ready
    protein_label = []
    for key in new_protein_label:
        protein_label.append(key 
                             + ' '*(len(top[-1])-len(key)-len(str(new_protein_label[key]))) 
                             + str(new_protein_label[key]))

    # build new top file with new protein label and correct force field
    new_top = ([f'#include "toppar/{file}"' for file in sorted(os.listdir(f'{di}/toppar/'))
             if file[:10] == 'martini_v3']
            + [f'#include "toppar/{file}"' for file in sorted(os.listdir(f'{di}/toppar/'))
               if file[:10] != 'martini_v3']
            + top[:top.index('[ molecules ]')+2]
            + protein_label
            + top[[top.index(line) for line in top if line[:4] == list(membrane_comp.keys())[0]][0]:])

    # write new top file
    with open(f'{di}/topol.top', 'w') as top_file:
        top_file.truncate()
        for line in new_top:
            top_file.write(line + '\n')

    input_log['protein name change in topol.top'] = '\n'.join(protein_label)
    save_log(di, input_log)
    
    ###############################
    # Prepare index file ##########
    ###############################
    
    # get input for index file creation
    membrane = ' | '.join(str(x) for x in np.arange(13,len(membrane_comp.keys())+13))
    solute = len(membrane_comp.keys()) + 16

    command = (f'{membrane}\n'
               f'name {len(membrane_comp.keys()) + 17} Membrane\n'
               f'{solute}\n'
               f'name {len(membrane_comp.keys()) + 18} Solute\n'
               f'\n'
               f'q\n')

    # make index file
    make_ndx = f'{gmx_index} -f {di}/run.gro -o {di}/index.ndx <<EOF\n{command}EOF'
    print(trun(make_ndx))

    input_log['index file'] = make_ndx
    save_log(di, input_log)
    
    ###############################
    # Do minimization #############
    ###############################
    
    log = []

    # copy gro-file into the simulation folder
    os.system(f'cp {di}/run.gro {min_dir}/minimized.gro')

    for num, step in enumerate(min_setup):

        # name
        step_name = step.split('/')[-1][:-4]

        # creating tpr file
        grompp = (f'{gmx_grompp} -f {step} -c {min_dir}/minimized.gro -p {di}/topol.top'
                  f' -o {min_dir}/{step_name}.tpr '
                  f' -n {di}/index.ndx')

        print(trun(grompp, verbose=True))
        sleep(5)
        
        # doing the minimization step
        mdrun = (f'{gmx_mdrun} -deffnm {min_dir}/{step_name}')
        print(trun(mdrun, verbose=True))

        log.append('\n'.join([f'step{num} = '+step,
                              'grompp = '+grompp,
                              'mdrun = '+mdrun]))
        os.system(f'cp {min_dir}/{step_name}.gro {min_dir}/minimized.gro')

    input_log['minimization steps'] = '\n\n'.join(log)
    save_log(di, input_log)
    
    ###############################
    # Do equilibration ############
    ###############################    

    log = []

    # copy gro-file into the simulation folder
    os.system(f'cp {min_dir}/minimized.gro {equi_dir}/equilibrated.gro')

    for step in equi_setup:

        # name
        step_name = step.split('/')[-1][:-4]

        # creating tpr file
        grompp = (f'{gmx_grompp} -f {step} -c {equi_dir}/equilibrated.gro -p {di}/topol.top'
                  f' -o {equi_dir}/{step_name}.tpr -r {equi_dir}/equilibrated.gro'
                  f' -n {di}/index.ndx')
        print(trun(grompp))
        sleep(5)
        
        # doing the equilibration step
        mdrun = (f'{gmx_mdrun} -deffnm {equi_dir}/{step_name}')
        print(trun(mdrun))

        log.append('\n'.join([f'step{num} = '+step,
                              'grompp   = '+grompp,
                              'mdrun = '+mdrun]))
        os.system(f'cp {equi_dir}/{step_name}.gro {equi_dir}/equilibrated.gro')

    input_log['equilibration steps'] = '\n\n'.join(log)
    save_log(di, input_log)
    
    ###############################
    # generate inital path ########
    ############################### 

    # early setup
    copy(mdp_path, di)

    # run parameters
    grompp = (f'{gmx_grompp} -f {di}/run.mdp -c {equi_dir}/equilibrated.gro -p {di}/topol.top'
              f' -o {initial_dir}/prun.tpr'
              f' -n {di}/index.ndx')
    mdrun = (f'{gmx_mdrun} -deffnm {initial_dir}/prun')
    
    log_tran = '\n'.join(['stateA      = '+str(stateA),
                          'stateB      = '+str(stateB),
                          'check_every = '+str(check_every),
                          'grompp      = '+grompp,
                          'mdrun       = '+mdrun])

    # creating tpr file
    print(trun(grompp, verbose=True))
    sleep(5)
    
    # doing the minimization step
    print(trun(f'{mdrun} -nsteps 1', verbose=True))
    
    # prepare
    global frame, index, traj_con
    topology = f'{initial_dir}/prun.gro'
    frame = md.load(topology)
    input_log['generate transition'] = log_tran
    save_log(di, input_log)

    # main loop
    subtrajectory = f'{initial_dir}/prun.trr'
    index = 0
    with open(f'{initial_dir}/drmsd_control', "a+", encoding="utf-8") as meta_txt:
        meta_txt.seek(0)
        meta_txt.truncate()
    traj_con = np.empty(shape=[0, 2])

    while not stop(subtrajectory, stateA=stateA, drmsd_reference=drmsd_reference):
        print(traj_con, index)
        with open(f'{initial_dir}/drmsd_control', "a+", encoding="utf-8") as meta_txt:
            for line in traj_con:
                meta_txt.write(f'{line[0]} {line[1]}\n')
        traj_con = np.empty(shape=[0, 2])
        process = (f'{mdrun} -cpi {initial_dir}/prun.cpt -nsteps -1 -maxh {check_every/3600}')
        cancel(process, verbose=False)
        print(trun(process))
        sleep(1)

    print(trun(process))
    
    ###############################
    # cutout inital path ########
    ############################### 
    
    # load trajectory
    traj = md.load(f'{initial_dir}/prun.trr', top=f'{initial_dir}/prun.gro')
    drmsd = DRMSD(traj, drmsd_reference)

    temp = []
    for i in drmsd:
        if i < stateA:
            temp.append(i)
            if np.max(temp) > stateB and np.min(temp) < stateA:
                break
            temp = [i]
        if i <= stateB and i >= stateA:
            temp.append(i)
        if i > stateB:
            temp.append(i)
            if np.max(temp) > stateB and np.min(temp) < stateA:
                break
            temp = [i]
            
    if np.max(temp) > stateB and np.min(temp) < stateA:
        initial_stats = (f'max : {np.round(np.max(temp), 3)} CV\n'
                         f'min : {np.round(np.min(temp), 3)} CV\n'
                         f'time: {len(temp)*traj.timestep*0.001} ns')
        print(initial_stats)
        traj[np.isin(drmsd, temp)].save_trr(f'{initial_dir}/initial.trr')

        input_log['initial trajectory stats'] = initial_stats
        save_log(di, input_log)
    else:
        print(f'No transition in CV range {stateA}-{stateB}')
    
    ###############################
    # create aimmd folder #########
    ############################### 
    
    aimmd = mkdir(f'{di}/aimmd')
    
    # copy all relevant files to aimmd folder
    os.system(f'cp -r {aimmd_params_path}/* {aimmd}')
    os.system(f'cp -r {di}/toppar {aimmd}')
    os.system(f'cp -r {initial_dir}/initial.trr {aimmd}')
    os.system(f'cp -r {initial_dir}/prun.gro {aimmd}/run.gro')
    os.system(f'cp -r {di}/run.mdp {aimmd}')
    os.system(f'cp -r {di}/topol.top {aimmd}')
    os.system(f'cp -r {di}/index.ndx {aimmd}')
    os.system(f'cp -r {aimmd_params_path}/../ire1_dimer_drmsd_ref/connections* {aimmd}')

    prun_cluster = mkdir(f'{aimmd}/prun_cluster')
    os.system(f'cp -r {aimmd}/params.py {prun_cluster}')
    os.system(f'cp -r {aimmd}/initial.trr {prun_cluster}')



def make_new_initial(params_path:str='bilayer_params.py', new_params:dict={}):

    ###############################
    # Get correct params ##########
    ###############################

    # get default params
    exec(open(params_path, 'r').read(), globals())
    # get new params
    if type(new_params) == dict:
        for value in new_params:
            exec(f'{value} = {new_params[value]}', globals())
    else:
        print(f'new_params has to be type dict, you provided {type(new_params)}')


    ###############################
    # Set up directories ##########
    ###############################

    # make new directory
    systems = mkdir(systems_path)
    name  = '_'.join([f'{key}{membrane_comp[key]}'for key in membrane_comp])
    mcomp = ' '.join([f'-l {key}:{membrane_comp[key]}'for key in membrane_comp])

    # making input log file
    input_log = {'pdb':pdb,
                 'membrane composition':name,
                 'xyz':f'{x} {y} {z}',
                 'salt': salt,
                 'toppar_path': toppar_path}

    # make new directory
    di = mkdir(f'{systems}/{name}')
    equi_dir = mkdir(f'{di}/equilibration')
    min_dir = mkdir(f'{di}/minimization')
    initial_dir = mkdir(f'{di}/initial')
    os.system(f'rm -rf {di}/toppar')
    os.system(f'cp -r {toppar_path} {di}/toppar')

    ###############################
    # generate inital path ########
    ###############################

    # early setup
    copy(mdp_path, di)

    # run parameters
    grompp = (f'{gmx_grompp} -f {di}/run.mdp -c {equi_dir}/equilibrated.gro -p {di}/topol.top'
              f' -o {initial_dir}/prun.tpr'
              f' -n {di}/index.ndx')
    mdrun = (f'{gmx_mdrun} -deffnm {initial_dir}/prun')

    log_tran = '\n'.join(['stateA      = '+str(stateA),
                          'stateB      = '+str(stateB),
                          'check_every = '+str(check_every),
                          'grompp      = '+grompp,
                          'mdrun       = '+mdrun])

    # creating tpr file
    print(trun(grompp, verbose=True))
    sleep(5)

    # doing the minimization step
    print(trun(f'{mdrun} -nsteps 1', verbose=True))

    # prepare
    global frame, index, traj_con
    topology = f'{initial_dir}/prun.gro'
    frame = md.load(topology)
    input_log['generate transition'] = log_tran
    save_log(di, input_log)

    # main loop
    subtrajectory = f'{initial_dir}/prun.trr'
    index = 0
    with open(f'{initial_dir}/drmsd_control', "a+", encoding="utf-8") as meta_txt:
        meta_txt.seek(0)
        meta_txt.truncate()
    traj_con = np.empty(shape=[0, 2])

    while not stop(subtrajectory, stateA=stateA, drmsd_reference=drmsd_reference):
        print(traj_con, index)
        with open(f'{initial_dir}/drmsd_control', "a+", encoding="utf-8") as meta_txt:
            for line in traj_con:
                meta_txt.write(f'{line[0]} {line[1]}\n')
        traj_con = np.empty(shape=[0, 2])
        process = (f'{mdrun} -cpi {initial_dir}/prun.cpt -nsteps -1 -maxh {check_every/3600}')
        cancel(process, verbose=False)
        print(trun(process))
        sleep(1)

    print(trun(process))

    ###############################
    # cutout inital path ##########
    ###############################

    # load trajectory
    traj = md.load(f'{initial_dir}/prun.trr', top=f'{initial_dir}/prun.gro')
    drmsd = DRMSD(traj, drmsd_reference)

    temp = []
    for i in drmsd:
        if i < stateA:
            temp.append(i)
            if np.max(temp) > stateB and np.min(temp) < stateA:
                break
            temp = [i]
        if i <= stateB and i >= stateA:
            temp.append(i)
        if i > stateB:
            temp.append(i)
            if np.max(temp) > stateB and np.min(temp) < stateA:
                break
            temp = [i]

    if np.max(temp) > stateB and np.min(temp) < stateA:
        initial_stats = (f'max : {np.round(np.max(temp), 3)} CV\n'
                         f'min : {np.round(np.min(temp), 3)} CV\n'
                         f'time: {len(temp)*traj.timestep*0.001} ns')
        print(initial_stats)
        traj[np.isin(drmsd, temp)].save_trr(f'{initial_dir}/initial.trr')

        input_log['initial trajectory stats'] = initial_stats
        save_log(di, input_log)
    else:
        print(f'No transition in CV range {stateA}-{stateB}')
        return

    ###############################
    # create aimmd folder #########
    ###############################

    aimmd = mkdir(f'{di}/aimmd')

    # copy all relevant files to aimmd folder
    os.system(f'cp -r {aimmd_params_path}/* {aimmd}')
    os.system(f'cp -r {di}/toppar {aimmd}')
    os.system(f'cp -r {initial_dir}/initial.trr {aimmd}')
    os.system(f'cp -r {initial_dir}/prun.gro {aimmd}/run.gro')
    os.system(f'cp -r {di}/run.mdp {aimmd}')
    os.system(f'cp -r {di}/topol.top {aimmd}')
    os.system(f'cp -r {di}/index.ndx {aimmd}')
    os.system(f'cp -r {aimmd_params_path}/../ire1_dimer_drmsd_ref/connections* {aimmd}')

    prun_cluster = mkdir(f'{aimmd}/prun_cluster')
    os.system(f'cp -r {aimmd}/params.py {prun_cluster}')
    os.system(f'cp -r {aimmd}/initial.trr {prun_cluster}')


def set_up_aimmd(di:str, aimmd_path:str, drmsd_path:str='../ire1_dimer_drmsd_ref'):
    ###############################
    # create aimmd folder #########
    ###############################

    aimmd = mkdir(f'{di}/aimmd')

    # copy all relevant files to aimmd folder
    os.system(f'cp -r {aimmd_params_path}/* {aimmd}')
    os.system(f'cp -r {di}/toppar {aimmd}')
    os.system(f'cp -r {di}/initial/initial.trr {aimmd}')
    os.system(f'cp -r {di}/initial/prun.gro {aimmd}/run.gro')
    os.system(f'cp -r {di}/run.mdp {aimmd}')
    os.system(f'cp -r {di}/topol.top {aimmd}')
    os.system(f'cp -r {di}/index.ndx {aimmd}')
    os.system(f'cp -r {drmsd_path}/connections* {aimmd}')

    prun_cluster = mkdir(f'{aimmd}/prun_cluster')
    os.system(f'cp -r {aimmd}/params.py {prun_cluster}')
    os.system(f'cp -r {aimmd}/initial.trr {prun_cluster}')
