import os
import sys
import cv2
import glob
import time
import select
import pickle
import pathlib
import colorsys
import warnings
import threading
import subprocess
import numpy as np
import mdtraj as md
import pandas as pd
import nglview as nv
from tqdm import tqdm
from time import sleep
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mdtraj.geometry.distance import _reduce_box_vectors

def trun(command:str, open_terminal:bool=False, stay_open:(int, str)=10, timeout=int(1e9), verbose=False):
    
    """
    Executes a shell command in different modes, capturing output or running in a terminal.

    Parameters
    ----------
    command : str
        The shell command to execute.
    open_terminal : bool, optional
        If True, opens a new terminal window to run the command. Default is False.
    stay_open : int or str, optional
        Time in seconds for the terminal to stay open if `open_terminal` is True. Default is 10.
    timeout : int, optional
        Maximum time in seconds to wait for command completion in threaded execution. Default is 1e9.
    verbose : bool, optional
        If True, prints the command output in real-time. Default is False.
    normal : bool, optional
        If True, executes the command in normal blocking mode and returns the output as a string.
        Default is True.

    Returns
    -------
    str or None
        If `normal` is True, returns the captured output of the command as a string.
        If `verbose` is True and `normal` is False, prints the output to the console.
        Otherwise, returns None.

    Notes
    -----
    - If `open_terminal` is True, the command is executed in a new terminal window with a pause for `stay_open` seconds.
    - If `normal` is False and `open_terminal` is False, the command runs in a separate thread with optional timeout.
    - Handles capturing both stdout and stderr in real-time.
    - If the command exceeds the timeout, the process is killed, and a message is appended to the output.
    - This function is useful for executing shell commands from Python scripts while optionally capturing or displaying output.
    """
    
    if open_terminal == True:
        os.system(f'gnome-terminal --disable-factory -- bash -c \"{command}; sleep {stay_open}\" ')
        
    else:
        txt = []

        def target():
            global proc
            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    preexec_fn=os.setsid,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True
                )


                while True:
                    reads = [proc.stdout.fileno(), proc.stderr.fileno()]
                    ret = select.select(reads, [], [])
                    for fd in ret[0]:
                        if fd == proc.stdout.fileno():
                            read = proc.stdout.readline()
                            if read:
                                txt.append(read)
                        if fd == proc.stderr.fileno():
                            read = proc.stderr.readline()
                            if read:
                                txt.append(read)
                    if proc.poll() is not None:
                        break

                proc.stdout.close()
                proc.stderr.close()
            except Exception as e:
                txt.append(f'Error executing {command}: {e}')
                print(f'Error executing {command}: {e}')

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            txt.append(f'echo "Timeout reached, killing process..."')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
            thread.join()

        # Check for return code
        if proc.returncode != 0:
            txt.append(f'echo "Return code: {proc.returncode}"')
        if verbose == True:
            print(''.join(txt))
            return ''.join(txt)
        else:
            return None

def mkdir(direc:str, verbose=False):
    if not os.path.exists(direc):
        os.makedirs(direc)
    elif verbose:
        print('There is already a directory')
    else:
        pass
    return direc

def pkl_save(path:(pathlib.Path, str), file):

    """
    Saves a Python object to a file using pickle serialization.

    Parameters
    ----------
    path : str
        Path to the file where the object should be saved.
    file : object
        Python object to be serialized and saved.

    Notes
    -----
    - Uses the highest available pickle protocol for efficiency and compatibility.
    - Overwrites the file if it already exists.
    - Can be used to save any picklable Python object such as dictionaries, lists, or custom objects.
    """
    
    with open(path, 'wb') as pkl:
        pickle.dump(file, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        
def pkl_load(path:(pathlib.Path, str), verbose=True):
    
    """
    Loads Python object(s) from file(s) using pickle serialization.

    Parameters
    ----------
    path : str
        Path or glob pattern to the file(s) to load.

    Returns
    -------
    object or dict
        - If a single file matches the path, returns the unpickled Python object.
        - If multiple files match and end with '.h5', returns a dictionary where
          keys are filenames (without '.h5') and values are the unpickled objects.

    Notes
    -----
    - Prints a warning if no files are found.
    - Ignores files that do not have the '.h5' extension when multiple matches exist.
    - Can be used to load any picklable Python object such as dictionaries, lists, or custom objects.
    """
    
    if isinstance(path, pathlib.Path):
        path = str(path)
    files = glob.glob(path)
    if len(files) == 0 and verbose:
        print(f'Can not find any file with this path: {path}')
    elif len(files) == 1:
        with open(path, 'rb') as pkl:
            return pickle.load(pkl)
    else:
        all_files = {}
        for file in files:
            if not file[-3:] == '.h5':
                pass
            else:
                with open(file, 'rb') as pkl:
                    all_files[file.split('/')[-1][:-3]] = pickle.load(pkl)
        return all_files

def copy(paths:(str, list), save:str, backup:bool=False, verbose=False):
    _backup = ''
    if backup == True:
        _backup = '--backup'
    if type(paths) == list:
        _dir = []
        _file = []
        for path in paths:
            _dir.append(os.path.isdir(path))
            _file.append(os.path.isfile(path))
            if (_dir[-1] + _file[-1]) == 0:
                print(f'This path is not a file or directory:\n{path}')
                return
            os.popen(f'cp -r {path} {save} {_backup}')
        if verbose == True:
            print(f'number of files       transfered to {save}: {np.sum(_file)}')
            print(f'number of directories transfered to {save}: {np.sum(_dir)}')
    elif type(paths) == str:
        _dir = os.path.isdir(paths)
        _file = os.path.isfile(paths)
        if (_dir + _file) == 0:
            print(f'This path is not a file or directory:\n{paths}')
            return
        os.popen(f'cp -r {paths} {save} {_backup}')
        if verbose == True:
            print(f'number of files       transfered to {save}: {np.sum(_file)}')
            print(f'number of directories transfered to {save}: {np.sum(_dir)}')
    else:
        print(f'Wrong file type: {type(paths)}\nuse str or list')

def cancel(command:str, verbose:bool=False):
    split = command.split()
    for i in os.popen(f'pgrep -a {split[0]}').read().split('\n'):
        if i.split()[1:] == split:
            os.system(f'kill {i.split()[0]}')
            if verbose == True:
                print(f'killed process: {i.split()[0]}')
                print(i)
            return
    if verbose == True:
        print('process is not running')
        
def DRMSD(trajectory:md.Trajectory, con:str='aimmd_params/connections_ref.npy'):
    Connections = np.load(con)
    d = md.compute_distances(trajectory,
                             Connections[:, :2].astype(int))
    d0 = Connections[:, 2]
    n = len(Connections)
    drmsd = np.sqrt(np.sum((d - d0) ** 2, axis=1) / (n-1))
    return drmsd

def save_log(direc:str, input_log:dict):
    temp = {}
    try:
        with open(f'{direc}/.input_log.h5', 'rb') as file:
            temp = pickle.load(file)
    except:
        pass
    for i in input_log:
        temp[i] = [time.ctime(), input_log[i]]
    with open(f'{direc}/.input_log.h5', 'wb') as file:
        pickle.dump(temp, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'{direc}/input.log', 'w') as log:
        for stuff in temp:
            log.write(f'{(len(stuff)+3)*"-"}\n|{stuff}|:\n{(len(stuff)+3)*"-"}\n{temp[stuff][1]}\n({temp[stuff][0]})\n\n\n')
            

def compute_com_dist(traj:md.Trajectory, monA_index:np.array, monB_index:np.array, only_com=False):
    """Only make molecules whole

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A mdtraj trajectory in which you want to calculate the distance between two center of masses
            
        monA_index : numpy array of shape (n_atoms, ) of the atom indecies you want to include
            The index of the atoms from which you want to calculate the center of mass (beginning from 0)
            
        monB_index : numpy array of shape (n_atoms, ) of the atom indecies you want to include
            The index of the atoms from which you want to calculate the center of mass (beginning from 0)
            

        --------
        """
    
    sorted_bonds = np.empty(shape=[0,2], dtype='int32')
    for i in monA_index:
        sorted_bonds = np.append(sorted_bonds, np.array([[i, i+1]], dtype='int32'), axis=0)
    for i in monB_index:
        sorted_bonds = np.append(sorted_bonds, np.array([[i, i+1]], dtype='int32'), axis=0)
    
    traj = traj.make_molecules_whole(sorted_bonds=sorted_bonds, inplace=False)
    
    monA = md.compute_center_of_mass(traj.atom_slice(monA_index))
    monB = md.compute_center_of_mass(traj.atom_slice(monB_index))
    if only_com == True:
        return np.asanyarray([monA, monB], dtype=float)
    
    box = traj.unitcell_vectors
    orthogonal = np.allclose(traj.unitcell_angles, 90)
    box_vectors = box.transpose(0, 2, 1)
    out = np.empty((traj.xyz.shape[0],), dtype=float)
    
    for i in range(len(traj.xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)
        
        r12 = monA[i] - monB[i]
        r12 -= bv3*np.round(r12[2]/bv3[2]);
        r12 -= bv2*np.round(r12[1]/bv2[1]);
        r12 -= bv1*np.round(r12[0]/bv1[0]);
        dist = np.linalg.norm(r12)
        if not orthogonal:
            for ii in range(-1, 2):
                v1 = bv1*ii
                for jj in range(-1, 2):
                    v12 = bv2*jj + v1
                    for kk in range(-1, 2):
                        new_r12 = r12 + v12 + bv3*kk
                        dist = np.min(dist, np.linalg.norm(new_r12))
        out[i] = dist
    return out

