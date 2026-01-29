import pathlib
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import itertools
import torch
import os
import mdtraj as md
import MDAnalysis as mda
from functions import pkl_save, pkl_load
from tqdm import tqdm
from scipy.special import logit, expit
# from utils import evaluate
import matplotlib.pyplot as plt
import matplotlib.animation as animationt
from IPython import display
from time import sleep
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import helix_analysis as hel
import utils as aimmd

def get_trainingsdata(pathensemble, initial_path=None, verbose=False,
        keys=None):
    
    """
    Extracts training data from a pathensemble for use in machine learning.

    Parameters
    ----------
    pathensemble : object
        An object representing the collection of paths between states, used attributes and methods:
            - initial_states
            - internal_states
            - final_states
            - values(keys, internal, backward, forward)
            - descriptors(keys, internal, backward, forward)
            - update_values()
            - frame_descriptors
            - frame_states
    keys : array-like, optional
        Specific indices of paths to include. If None, all accepted paths are used.
    initial_path : object, optional
        Optional reference object providing descriptors for internal states A and B.
    verbose : bool, optional
        If True, prints progress and diagnostic information.

    Returns
    -------
    results : np.ndarray
        Array of shape (N, 2) with "shooting results" per path or frame, giving probabilities of reaching states A and B.
    descriptors : np.ndarray
        Array of feature descriptors for each path or frame.
    selection_probabilities : np.ndarray
        Array of selection probabilities for each path or frame, normalized to sum to 1.

    Notes
    -----
    - Values and descriptors are collected for forward and backward trajectories within the path ensemble.
    - Paths that start or end in R are binned to uniformize selection probabilities across the range of values.
    - Internal state frames (A and B) are incorporated with fixed probabilities.
    - The function returns combined arrays for all states (A, B, R) suitable for training supervised models of transitions.
    """
   
    smoothening_weight = 1e-3
    regularization_weight = 1e-3
    stop = 30.
    nbins = 10
   
    pathensemble.update_values()  # with previous model

    t0 = time.time()
    losses, scales, selection_probabilities, results = [], [], [], []

    if len(pathensemble):
        keys = np.arange(len(pathensemble))[keys].ravel()
        keys = keys[pathensemble[keys].are_accepted]
    else:
        keys = np.zeros(0, dtype=int)

    # extract info
    initial_states = pathensemble.initial_states[keys]
    internal_states = pathensemble.internal_states[keys]
    final_states = pathensemble.final_states[keys]

    # step 1. paths in R
    # which backward and forward paths belong to A and B?
    backwardA = keys[np.where(
        (internal_states == 'R') * (initial_states == 'A'))[0]]
    backwardB = keys[np.where(
        (internal_states == 'R') * (initial_states == 'B'))[0]]
    forwardA = keys[np.where(
        (internal_states == 'R') * (final_states == 'A'))[0]]
    forwardB = keys[np.where(
        (internal_states == 'R') * (final_states == 'B'))[0]]

    # get values and descriptors for A
    if len(forwardA) and len(backwardA):
        valuesA = np.append(
            np.concatenate(pathensemble.values(forwardA,
                internal=True, backward=False, forward=True)),
            np.concatenate(pathensemble.values(backwardA,
                internal=True, backward=True, forward=False)))
        descriptorsA = np.append(
            np.concatenate(pathensemble.descriptors(forwardA,
                internal=True, backward=False, forward=True), axis=0),
            np.concatenate(pathensemble.descriptors(backwardA,
                internal=True, backward=True, forward=False), axis=0),
            axis=0)
    elif len(forwardA):
        valuesA = np.concatenate(pathensemble.values(forwardA,
            internal=True, backward=False, forward=True))
        descriptorsA = np.concatenate(pathensemble.descriptors(forwardA,
            internal=True, backward=False, forward=True), axis=0)
    elif len(backwardA):
        valuesB = np.concatenate(pathensemble.values(backwardA,
                internal=True, backward=True, forward=False))
        descriptorsB = np.concatenate(pathensemble.descriptors(backwardA,
                internal=True, backward=True, forward=False), axis=0)
    else:
        valuesA = np.zeros(0)
        try:
            dim = len(initial_path.frame_descriptors[0])
        except:
            dim = len(pathensemble.frame_descriptors[0])
        descriptorsA = np.zeros((0, dim))

    # get values and descriptors for B
    if len(forwardB) and len(backwardB):
        valuesB = np.append(
            np.concatenate(pathensemble.values(forwardB,
                internal=True, backward=False, forward=True)),
            np.concatenate(pathensemble.values(backwardB,
                internal=True, backward=True, forward=False)))
        descriptorsB = np.append(
            np.concatenate(pathensemble.descriptors(forwardB,
                internal=True, backward=False, forward=True), axis=0),
            np.concatenate(pathensemble.descriptors(backwardB,
                internal=True, backward=True, forward=False), axis=0),
            axis=0)
    elif len(forwardB):
        valuesB = np.concatenate(pathensemble.values(forwardB,
            internal=True, backward=False, forward=True))
        descriptorsB = np.concatenate(pathensemble.descriptors(forwardB,
            internal=True, backward=False, forward=True), axis=0)
    elif len(backwardB):
        valuesB = np.concatenate(pathensemble.values(backwardB,
                internal=True, backward=True, forward=False))
        descriptorsB = np.concatenate(pathensemble.descriptors(backwardB,
                internal=True, backward=True, forward=False), axis=0)
    else:
        valuesB = np.zeros(0)
        try:
            dim = len(initial_path.frame_descriptors[0])
        except:
            dim = len(pathensemble.frame_descriptors[0])
        descriptorsB = np.zeros((0, dim))

    # uniformize in bins
    bins = np.array([-np.inf, +np.inf])
    if len(valuesA) and len(valuesB):
        vmin = np.min(valuesB)
        vmax = np.max(valuesA)
        if vmax - vmin > 1.5:
            bins = np.concatenate(
                [[-np.inf],
                np.linspace(vmin, vmax, round(vmax - vmin) + 1)[1:-1],
                [+np.inf]])
    indicesA = np.digitize(valuesA, bins)
    indicesB = np.digitize(valuesB, bins)
    selection_probabilitiesA = np.ones(len(valuesA))  # in batch
    selection_probabilitiesB = np.ones(len(valuesB))
    resultsA = np.zeros((len(valuesA), 2))  # "shooting results"
    resultsB = np.zeros((len(valuesB), 2))
    # SOLUTION: sA * wA = const within each bin

    for i in range(len(bins) + 1):
        keepersA = np.where(indicesA == i)[0]
        keepersB = np.where(indicesB == i)[0]
        nA = len(keepersA)
        nB = len(keepersB)
        if not nA + nB:
            continue

        # inversely proportional selection
        if nA:
            selection_probabilitiesA[keepersA] = 1 / nA
        if nB:
            selection_probabilitiesB[keepersB] = 1 / nB
        resultsA[keepersA, 0] = nA / (nA + nB)
        resultsB[keepersB, 1] = nB / (nA + nB)

    # step 2: include data inside states
    internalA = keys[np.where(internal_states == 'A')[0]]
    internalB = keys[np.where(internal_states == 'B')[0]]
    if not len(internalA):
        internal_descriptorsA = initial_path.frame_descriptors[
            initial_path.frame_states == 'A']
    else:
        internal_descriptorsA = np.concatenate(
            pathensemble.descriptors(internalA,
            internal=True, backward=False, forward=True), axis=0)
    if not len(internalB):
        internal_descriptorsB = initial_path.frame_descriptors[
            initial_path.frame_states == 'B']
    else:
        internal_descriptorsB = np.concatenate(
            pathensemble.descriptors(internalB,
            internal=True, backward=False, forward=True), axis=0)

    # put all together
    nA = len(descriptorsA)
    nB = len(descriptorsB)
    nA0 = len(internal_descriptorsA)
    nB0 = len(internal_descriptorsB)

    # total by states
    selection_probabilitiesA = np.append(np.repeat(1 / nA0, nA0),
                                         selection_probabilitiesA)
    selection_probabilitiesB = np.append(np.repeat(1 / nB0, nB0),
                                         selection_probabilitiesB)
    resultsA = np.append(
        np.repeat([[1., 0.]], nA0, axis=0), resultsA, axis=0)
    resultsB = np.append(
        np.repeat([[0., 1.]], nB0, axis=0), resultsB, axis=0)
    valuesA = np.append(np.repeat(-np.inf, nA0), valuesA)
    valuesB = np.append(np.repeat(+np.inf, nB0), valuesB)
    descriptorsA = np.append(internal_descriptorsA, descriptorsA, axis=0)
    descriptorsB = np.append(internal_descriptorsB, descriptorsB, axis=0)

    # all states
    selection_probabilities = np.append(selection_probabilitiesA,
                                        selection_probabilitiesB)
    selection_probabilities /= np.sum(selection_probabilities)
    results = np.append(resultsA, resultsB, axis=0)
    values = np.append(valuesA, valuesB)
    descriptors = np.append(descriptorsA, descriptorsB, axis=0)

    return results, descriptors, selection_probabilities

class Network(torch.nn.Module):
    
    def __init__(self,
                 n_features:int= 3136,
                 hidden_layers:list[int]=[512],
                 activation=[torch.nn.PReLU(512)],
                 dropout:float=0.0,
                 batch_norm:bool=False):

        self.input_parameters = {'n_features'   : n_features,
                           'hidden_layers': hidden_layers,
                           'activation'   : activation,
                           'dropout'      : dropout,
                           'batch_norm'   : batch_norm}
        super().__init__()
        
        layers = []
        prev_size = n_features  # Input dimension (position)
        
        for j, size in enumerate(hidden_layers):
            
            # layers
            layers.append(torch.nn.Linear(prev_size, size))
            
            # Batchnorm
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(size))
            
            # activation 
            layers.append(activation[j])
            
            # drop out layer
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
                
            prev_size = size
            
        # Output layer (scalar free energy)
        layers.append(torch.nn.Linear(prev_size, 1))
        
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy at given positions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Positions to evaluate free energy at, shape [..., 1]
            
        Returns:
        --------
        torch.Tensor:
            Free energy values, shape [...]
        """
            
        return self.net(x)
    
class ZMatrix(Dataset):
    def __init__(self, data_paths, membrane_compositions = None, verbose=True):
        """
        Args:
            npz_file (.npz-file): .npz-file containing descriptors, results, and weights 
                                  from shootings, equilibriumA and equilibriumB conformations.
        """
        # Load the .npz file
        self.data_path             = data_paths
        self.membrane_compositions = membrane_compositions
        self.get(data_paths=data_paths,
                 membrane_compositions=membrane_compositions,
                 verbose=verbose)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.descriptors)
    
    def __getitem__(self, idx):
        """
        Returns a single sample of data.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (features, label, weight) for the given index.
        """
        descriptors = self.descriptors[idx]
        results = self.results[idx]
        return descriptors, results
    
    def get(self, data_paths, membrane_compositions=None, verbose=True):
        """
        Returns a single sample of data.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (features, label, weight) for the given index.
        """
        self.data_paths             = data_paths
        self.membrane_compositions = membrane_compositions
        
        setup = True
        for path in tqdm(data_paths,
                         total = len(data_paths),
                         disable= not verbose):
            
            if membrane_compositions:
                if path.stem not in membrane_compositions:
#                     print(path.stem)
                    continue
                    
            dataset = np.load(path)
            mask = np.invert((dataset['results'][:, 0] == 1) + (dataset['results'][:, 1] == 1))
            if setup:
                descriptors             = np.empty((0, dataset['descriptors'].shape[-1]))
                results                 = np.empty((0, dataset['results'].shape[-1]))
                selection_probabilities = np.empty((0, ))
                setup = False
                
            descriptors = np.append(descriptors, dataset['descriptors'], axis=0)
            results = np.append(results, dataset['results'], axis=0)
            selection_probabilities = np.append(selection_probabilities,
                                                dataset['selection_probabilities'])
            
            

        self.descriptors = descriptors
        self.results = results
        self.selection_probabilities = selection_probabilities
        return 


def convert_mdanalysis_to_mdtraj(trajectory, topology):
    
    """
    Convert an MDAnalysis trajectory or list of trajectories to an MDTraj trajectory.

    This function takes a trajectory (or list of trajectories) from MDAnalysis and
    converts it into an MDTraj `Trajectory` object. It copies atomic positions,
    unit cell vectors, and frame times, adjusting units from angstroms to nanometers.

    Parameters
    ----------
    trajectory : MDAnalysis.Universe or list of MDAnalysis.Universe
        A single MDAnalysis trajectory or a list of trajectory frames to convert.
    topology : str or mdtraj.Topology
        Path to a topology file (e.g., GRO, PDB) or an MDTraj Topology object used
        to initialize the MDTraj trajectory.

    Returns
    -------
    mdtraj.Trajectory
        An MDTraj `Trajectory` object containing the same frames, positions, and
        unit cell vectors as the original MDAnalysis trajectory.

    Notes
    -----
    - Positions and unit cell vectors are converted from angstroms to nanometers.
    - Handles both single trajectories and lists of trajectory objects.
    - Frames that cannot be accessed are skipped silently.
    - The function reshapes positions and unit cell vectors to match MDTraj's expected
      input format.
    """
        
    position_array = []
    unitcell_vectors_array = []
    time_array = []
    new_traj = md.load(topology)
    for frame in trajectory:
        positions = frame.positions
        position_array.append(positions.reshape((-1, *positions.shape)) / 10.)
        unitcell_vectors_array.append(frame.triclinic_dimensions.reshape((-1, 3, 3)) / 10.)
        time_array.append(frame.time)
    new_traj.xyz = np.vstack(position_array)
    new_traj.unitcell_vectors = np.vstack(unitcell_vectors_array)
    new_traj.time = np.asanyarray(time_array)
    return new_traj


def descriptors_lipid_heads(trajectory, atom_to_center_monA=101, monB=101+187):
    
    """
    Compute descriptors for lipid head positions relative to a reference atom.

    This function centers the trajectory on a specified reference atom (monA) and
    extracts the positions of the first beads (heads) of lipids. It returns
    the positions of the 100 nearest lipid heads to the reference atom along with
    their types and the protein coordinates of another reference atom (monB).

    Args:
        trajectory (md.core.trajectory.Trajectory):
            The trajectory to extract descriptors from.
        atom_to_center_monA (int, optional):
            Index of the atom used to center the trajectory. Default is 101.
        monB (int, optional):
            Index of the protein atom whose coordinates are included. Default is 288 (101 + 187).

    Returns:
        tuple:
            - np.ndarray: Array of shape (n_frames, 402) containing protein coordinates
              and flattened lipid head descriptors (x, y, z, type) for each frame.
            - list: List of lipid types and their corresponding indices.
    """

    traj1 = center_mdtraj(trajectory, atom_to_center=atom_to_center_monA)
    beads  = [i.residue.name for i in list(traj1.topology.atoms)]
    lipids = [i.name for i in list(traj1.topology.residues) if len(i.name) == 4]
    lipid_first_bead = []
    lipid_types = []
    for j, i in enumerate(np.unique(lipids)):
        lipid_first_bead.append([i, list(traj1.topology.atoms)[beads.index(i)].name])
        lipid_types.append([i, j])
    lipid_first_bead_index = np.asanyarray([[i.index,
                                             np.where(i.residue.name==np.asanyarray(lipid_first_bead)[:, 0])[0][0]] 
                                            for i in list(traj1.topology.atoms) 
                              if [i.residue.name,i.name] in lipid_first_bead])
    x = traj1.xyz[:, lipid_first_bead_index[:, 0], 0]
    y = traj1.xyz[:, lipid_first_bead_index[:, 0], 1]
    ref = traj1.xyz[:, [atom_to_center_monA], :2]
    r = np.argsort(((x-ref[:,:,0])**2+(y-ref[:,:,0])**2)**0.5, axis=1)[:, :100]
    x = np.empty((r.shape[0], r.shape[1], 1))
    y = np.empty((r.shape[0], r.shape[1], 1))
    z = np.empty((r.shape[0], r.shape[1], 1))
    lipid = np.empty((r.shape[0], r.shape[1], 1))
    for j,i in enumerate(r):
        x[[j], :100, 0]     = traj1.xyz[[j], lipid_first_bead_index[i, 0], 0]
        y[[j], :100, 0]     = traj1.xyz[[j], lipid_first_bead_index[i, 0], 1]
        z[[j], :100, 0]     = traj1.xyz[[j], lipid_first_bead_index[i, 0], 2]
        lipid[[j], :100, 0] = lipid_first_bead_index[i, 1]
    protein = traj1.xyz[:, monB, :2]
    lipid_heads = np.dstack((x,y,z,lipid))
    lipid_heads = np.reshape(lipid_heads, (lipid_heads.shape[0], 400))
    return np.hstack((protein,lipid_heads)), lipid_types

def center_mdtraj(trjajectory, atom_to_center:int=101):
    
    """
    Center a trajectory in its simulation box around a specific atom.

    This function translates all atoms in each frame of the trajectory such that
    the specified atom is centered in the box. Coordinates are wrapped within
    the periodic box after centering.

    Args:
        trjajectory (md.core.trajectory.Trajectory):
            The trajectory to be centered.
        atom_to_center (int, optional):
            Index of the atom to center the trajectory on. Default is 101.

    Returns:
        md.core.trajectory.Trajectory:
            The centered trajectory with coordinates wrapped within the box.
    """

    trjajectory.xyz -= trjajectory.unitcell_lengths[:, None, :] / 2
    trjajectory.xyz -= trjajectory.xyz[:, [atom_to_center], :]
    trjajectory.xyz += trjajectory.unitcell_lengths[:, None, :] / 2
    trjajectory.xyz %= trjajectory.unitcell_lengths[:, None, :]
    return trjajectory

def new_descriptors_function(trajectory,
                             convert_to_mdtraj:bool=True,
                             Connection_path:str='committor_check/connections.npy'):
    Connections = np.load(Connection_path)
    if convert_to_mdtraj:
        trajectory = convert_mdanalysis_to_mdtraj(trajectory)
    distances = md.compute_distances(trajectory, Connections[:, :2])
    lipid_monA, lipid_types = descriptors_lipid_heads(trajectory, 101, 101+187)
    lipid_monB, lipid_types = descriptors_lipid_heads(trajectory, 101+187, 101)
    return np.hstack((distances,lipid_monA, lipid_monB)), lipid_types

def prepare_network(network, optimizer=None, lr = 0.001, cuda = True, save_path = None):
    
    """
    Set up the network device and optimizer.
    
    parameters: 
        network: initialized network (torch.nn.Module)
        lr: learning rate, default 0.001 (float)
        cuda: cuda driver available, default False (bool)
    
    returns:
        network: initialized network (torch.nn.Module)
        optimizer: Adam optimizer (torch.optim)
        device: device for the network (torch.device)
        dtype: data type for the network (torch.dtype)
    """

    if cuda:
        network.to('cuda')
    if not optimizer:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    
    if save_path:
        # print network architecture to file
        with open(f'{save_path}/network.txt', 'w') as f:
            f.write(str(network))
            f.write(f'\n\nNumber of parameters: {sum(p.numel() for p in network.parameters())}')
            # optimizer info
            f.write(f'\n\nOptimizer: {optimizer}')
            f.write(f'\nLearning rate: {lr}')
            
    
    return network, optimizer, device, dtype


def train(network, train_dataset, validation, verbose=False,
        keys=None, save_memory=False, epochs=1000, lr=1e-3, save_to='best', batch_size=4096, scaler=None):
     
    """
    Train a neural network using a training dataset and evaluate on a validation set.

    Args:
        network (torch.nn.Module): Neural network to train.
        train_dataset (object): Dataset object with descriptors, results, and selection probabilities.
        validation (dict): Dictionary of validation datasets {name: (conformations, reference)}.
        verbose (bool, optional): If True, prints training progress.
        keys (array-like, optional): Subset of training data to use.
        save_memory (bool, optional): If True, processes descriptors on the fly to save memory.
        epochs (int, optional): Number of training epochs.
        lr (float, optional): Initial learning rate.
        save_to (str, optional): Base filename for saving best models.
        batch_size (int, optional): Number of samples per training batch.
        scaler (object, optional): Scaler object to transform descriptors before evaluation.

    Returns:
        tuple:
            - losses (list[float]): Training losses per batch.
            - validation_losses (list[float]): Maximum validation loss per epoch.
            - scales (list[float]): Maximum network output per batch.
    """
    
    # training parameters
    smoothening_weight = 1e-3
    regularization_weight = 1e-3
    stop = 30.
    nbins = 10

    # nn setup
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    """
    Training loop.
    """

    # setup
    losses = []
    scales = []
    min_loss_1 = min_loss_2 = np.inf
    validation_losses = []

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    for epoch in pbar:
        pbar.set_description('Training')

        for param_group in optimizer.param_groups:
            # slowly increase lr
            param_group['lr'] = lr * min(1, (epoch + 1) / (epochs / 20))

        # sample batch
        indices = np.random.choice(len(train_dataset.selection_probabilities),
                                   batch_size, p=train_dataset.selection_probabilities/np.sum(train_dataset.selection_probabilities))
        if save_memory:  # separately to save memory
            d = process_descriptors(descriptors[indices])
        else:
            d = train_dataset.descriptors[indices]
        d = torch.tensor(d, dtype=dtype, device=device)
        d.requires_grad = True
        r = torch.tensor(train_dataset.results[indices], dtype=dtype, device=device)

        # define loss function
        def closure():
            optimizer.zero_grad()
            q = network(d)

            exp_pos_q = torch.exp(+q[:, 0])
            exp_neg_q = torch.exp(-q[:, 0])
            toA_contrib = r[:, 0] * torch.log(1. + exp_pos_q)
            toB_contrib = r[:, 1] * torch.log(1. + exp_neg_q)
            loss = torch.sum((toA_contrib + toB_contrib) / torch.sum(r))

            # Compute the smoothness penalty
            q_grad = torch.autograd.grad(
                outputs=q.sum(), inputs=d, create_graph=True)[0]
            smoothness_loss = (torch.abs(q_grad) ** 2).mean()
            loss += smoothening_weight * smoothness_loss

            # Calculate L1 regularization
            l1_norm = sum(p.abs().sum() for p in network.parameters())

            # Combine original loss with L1 regularization term
            loss += regularization_weight * l1_norm
            loss.backward()
            return loss

        # update network
        network.train()
        loss = optimizer.step(closure)
        losses.append(float(loss))

        # report scales
        q = network(d)
        scales.append(max(float(torch.max(q)), -float(torch.min(q))))
        Range = float(torch.min(q)), float(torch.max(q))

        # TEST
        network.eval()
        error_list = []
        validation_loss = []
        for comp in validation:
            conformations, reference = validation[comp]
            conformations = np.squeeze(np.asanyarray(conformations))
            if scaler:
                estimate = expit(aimmd.evaluate(network,
                                                scaler.transform(conformations),
                                                batch_size=batch_size))
            else:
                estimate = expit(aimmd.evaluate(network,
                                                conformations,
                                                batch_size=batch_size))
            for j in np.arange(len(estimate)):
                error = np.abs(reference[j]-estimate[j])/(
                        reference[j] if reference[j] < .5 else 1 - reference[j])
                error_list.append(error)
            coef = np.polyfit(reference, estimate, 1)
            correction = np.abs(1-coef[0]) + np.abs(0-coef[1])/2
            validation_loss.append(np.mean(error_list) + correction)
    
        val_mean = np.mean(validation_loss)
        val_max = np.max(validation_loss)
        
        validation_losses.append(val_max)

        # save the model if the test loss is the lowest
        if val_mean < min_loss_1:
            min_loss_1 = val_mean
            if save_to:
                # update the best model
                pkl_save(f'{save_to}_mean.h5', [network.input_parameters, network.state_dict()])
        
        if val_max < min_loss_2:
            min_loss_2 = val_max
            if save_to:
                # update the best model
                pkl_save(f'{save_to}_max.h5', [network.input_parameters, network.state_dict()])



        pbar.set_postfix({'Epoch':epoch,
                          'Train loss':"%.3g" %losses[-1],
                          'validation loss':np.round(validation_losses[-1], 3),
                          'best validation loss': np.round(np.min(validation_losses), 3),
                          'scales':np.round(scales[-1], 3)}) 


    return losses, validation_losses, scales


def get_membrane_thickness(trajectory:md.core.trajectory.Trajectory):
    
    """
    Compute the average thickness of a lipid membrane from a molecular dynamics trajectory.

    The function calculates the z-coordinate separation between the upper and lower leaflets
    of each lipid type in the trajectory and returns the membrane thickness corresponding 
    to the lipid with the largest average separation. It also computes the standard deviation
    of the leaflet positions as a measure of membrane fluctuation.

    Args:
        trajectory (md.core.trajectory.Trajectory): MDTraj trajectory containing the membrane system.

    Returns:
        tuple:
            - membrane_thickness (np.ndarray): Array of thickness values per frame for the lipid with the largest separation.
            - membrane_std (np.ndarray): Array of standard deviations of upper and lower leaflet positions per frame.
    """

    trajectory = center_mdtraj(trajectory)
    lipids = [i.name for i in list(trajectory.topology.residues) if len(i.name) == 4]
    beads  = [i.residue.name for i in list(trajectory.topology.atoms)]
    lipid_first_bead = []
    for j, i in enumerate(np.unique(lipids)):
        lipid_first_bead.append([i, list(trajectory.topology.atoms)[beads.index(i)].name])
    membrane_thickness = 0
    for lipid in lipid_first_bead:
        check = [bool((i.name==lipid[1])*
                     (i.residue.name == lipid[0])) for i in trajectory.topology.atoms]
        z_value_lipid = trajectory.xyz[:, check, -1]
        upper_leavlet = np.asanyarray([[np.mean(a[m])] for a,m in zip(z_value_lipid,z_value_lipid > np.mean(z_value_lipid))])
        lower_leavlet = np.asanyarray([[np.mean(a[m])] for a,m in zip(z_value_lipid,z_value_lipid < np.mean(z_value_lipid))])
        upper_leavlet_sdt = np.asanyarray([[np.std(a[m])] for a,m in zip(z_value_lipid,z_value_lipid > np.mean(z_value_lipid))])
        lower_leavlet_sdt = np.asanyarray([[np.std(a[m])] for a,m in zip(z_value_lipid,z_value_lipid < np.mean(z_value_lipid))])
        membrane_thickness_lipid = np.abs(upper_leavlet-lower_leavlet)
        if np.mean(membrane_thickness_lipid) > np.mean(membrane_thickness):
            membrane_thickness = membrane_thickness_lipid
            membrane_std = np.mean(np.asanyarray([upper_leavlet_sdt, lower_leavlet_sdt]), axis=0)
    return membrane_thickness, membrane_std

def get_distances(trajectory:md.core.trajectory.Trajectory,
                      Connection_path:str=Path('DRMSD_reference/CV_connections/connections.npy')):
    
    """
    Compute pairwise distances for specified atom pairs in a molecular dynamics trajectory.


    The function loads a set of atom index pairs from a .npy file and computes the
    distances between these atom pairs for each frame in the trajectory using MDTraj.


    Args:
        trajectory (md.core.trajectory.Trajectory): MDTraj trajectory object containing the system.
        Connection_path (str or Path, optional): Path to a .npy file containing atom index pairs. Defaults to 'DRMSD_reference/CV_connections/connections.npy'.


    Returns:
        np.ndarray: Array of distances with shape (n_frames, n_pairs), 
        where n_frames is the number of trajectory frames and n_pairs is the number of atom pairs.
    """


    Connections = np.load(Connection_path)
    distances = md.compute_distances(trajectory, Connections[:, :2])
    return distances

def get_lipid_composition(trajectory:md.core.trajectory.Trajectory, train_on):
    
    """
    Compute the lipid composition of a trajectory relative to a specified set of lipids.

    This function calculates the fractional abundance of each lipid type 
    present in `train_on` within the trajectory and scales it by 10. 
    The result is repeated for each frame in the trajectory.

    Args:
        trajectory (md.core.trajectory.Trajectory): 
            MDTraj trajectory object containing lipid molecules.

        train_on (list of str): 
            List of lipid names to include in the composition calculation.

    Returns:
        np.ndarray: 
            Array of shape (n_frames, n_lipids) representing the composition 
            of specified lipids for each frame, scaled by 10.
    """

    lipids = [i.name for i in list(trajectory.topology.residues) if len(i.name) == 4]
    lipid_counts = np.unique(lipids, return_counts=True)
    composition = np.zeros(len(train_on))
    number_of_lipids = np.sum(lipid_counts[-1])
    for lipid, number in list(zip(*lipid_counts)):
        if lipid in train_on:
            order = np.argmax((lipid == np.asanyarray(train_on)))
            composition[order] = number/number_of_lipids*10
    return np.repeat(np.array([composition]), len(trajectory), axis=0)




def get_validation_dataset(directory:(str, pathlib.posixpath),
                           descriptor_fuction,
                           membrane_compositions:list= None,
                           verbose=True):
    
    # convert path into PosixPath if necessary
    if isinstance(directory, str):
        directory = Path(directory)
    validation_dict = {}
    for pkl_path in tqdm(directory.glob('**/committor_check/*pkl'),
                         total=len(list(directory.glob('**/committor_check/*pkl'))),
                         disable=not verbose):
        name = pkl_path.parts[-3]
        if membrane_compositions:
            if name not in membrane_compositions:
                continue
        descriptors = []
        reference   = []
        ref_results = pkl_load(str(pkl_path))
        for key in ref_results:
            try:
                trajectory = md.load(pkl_path.parent / 'ref_frames' / f'{key}.gro')
            except:
                continue
            reference.append(ref_results[key][3])
            descriptors.append(np.squeeze(descriptor_fuction(trajectory, name)))
        validation_dict[name] = [np.asanyarray(descriptors), reference]
        
    return validation_dict

    
def transfer_membrane_comp(membrane_composition:str, train_on:list=None):
    
    """
    Transfer membrane composition data into a standardized array format.

    This function maps a dictionary of membrane composition to a fixed-length array
    corresponding to the lipid types specified in `train_on`. Each element of the array
    represents the normalized proportion of a lipid type in the membrane.

    Args:
        membrane_composition (str):
            Path to or representation of the membrane composition data.

        train_on (list):
            List of lipid names that define the order of the output array.

    Returns:
        np.ndarray:
            Array of lipid composition fractions aligned with `train_on`.
    """

    membrane_composition = extract_membrane_composition(membrane_composition)
    composition = np.zeros(len(train_on))
    for lipid in membrane_composition:
        if lipid in train_on:
            order = np.argmax((lipid == np.asanyarray(train_on)))
            composition[order] = membrane_composition[lipid]/10
    return np.asanyarray(composition)

def extract_membrane_composition(membrane_composition:str):
    
    """
    Extract lipid composition from a formatted membrane composition string.

    The input string should contain lipid names and their percentages concatenated
    with underscores (e.g., "POPC80_CHOL20"). This function parses the string and
    returns a dictionary mapping each lipid type to its percentage.

    Args:
        membrane_composition (str):
            Membrane composition string with lipid names and percentages.

    Returns:
        dict:
            Dictionary where keys are lipid types (str) and values are percentages (float).
    """

    lipids = membrane_composition.split('_')
    composition = {}
    for lipid in lipids:
        if len(lipid) < 5:
            continue
        lipid_type = lipid[:4]
        percent = lipid[4:]
        composition[lipid_type] = float(percent)
    return composition



def make_whole_mdtraj(trajectory):
    
    """
    Make molecules whole in an MDTraj trajectory by reconnecting broken bonds.

    This function creates a copy of the input MDTraj `Trajectory` and ensures that
    all protein molecules are made whole, i.e., fragmented molecules across periodic
    boundaries are reconnected. It generates a list of consecutive bonds along the
    protein chain and uses `mdtraj.Trajectory.make_molecules_whole` to reconstruct
    the molecules.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        The MDTraj trajectory to process. Expected to contain protein atoms.

    Returns
    -------
    mdtraj.Trajectory
        A new MDTraj trajectory with protein molecules made whole, with the same
        frames as the input trajectory.

    Notes
    -----
    - Bonds are generated consecutively along the protein atom indices.
    - The operation is performed in-place on a copy of the trajectory.
    - Useful for trajectories with periodic boundary conditions where molecules
      may appear split across the simulation box.
    """
        
    new_trajectory = trajectory[:]
    sorted_bonds = np.empty(shape=[0,2], dtype='int32')
    for i in trajectory.topology.select('protein'):
        sorted_bonds = np.append(sorted_bonds, np.array([[i, i+1]], dtype='int32'), axis=0)

    new_trajectory.make_molecules_whole(sorted_bonds=sorted_bonds, inplace=True)
    return new_trajectory

def convert_mdtraj_to_mdanalysis(traj, topology):
    
    """
    Convert an MDTraj trajectory to an MDAnalysis Universe.

    This function takes an MDTraj trajectory and a corresponding topology file,
    and returns an MDAnalysis Universe with the trajectory data properly loaded.
    Coordinates are converted from nanometers (MDTraj) to angstroms (MDAnalysis).

    Args:
        traj (md.core.trajectory.Trajectory):
            The MDTraj trajectory object to convert.

        topology (str):
            Path to the topology file compatible with MDAnalysis.

    Returns:
        mda.core.universe.Universe:
            An MDAnalysis Universe containing the trajectory data.
    """

    universe = mda.Universe(topology)
    
    # Convert nm → Å for MDAnalysis
    coordinates = traj.xyz * 10.0  # shape: (n_frames, n_atoms, 3)
    unitcells = traj.unitcell_lengths * 10.0 if traj.unitcell_lengths is not None else None
    angles = traj.unitcell_angles if traj.unitcell_angles is not None else None
    reader = MemoryReader(coordinates, lengths=unitcells, angles=angles)
    universe.trajectory = reader

    return universe

def get_helix_analysis(universe,
                       topology,
                       helix_selection:(list, str)= ['name BB and resnum 500-520',
                                                     'name BB and resnum 525-556'],
                       verbose=True):
    
    """
    Analyze helical segments in a protein trajectory.

    This function performs helix analysis on the specified segments of a protein
    trajectory. It supports input as either an MDTraj trajectory or an MDAnalysis
    Universe. The trajectory is first processed to remove jumps across periodic
    boundaries before analysis. Results are returned as a dictionary.

    Args:
        universe (md.core.trajectory.Trajectory or mda.core.universe.Universe):
            The trajectory or universe object containing the protein.

        topology (str):
            Topology file path used for MDAnalysis conversion if needed.

        helix_selection (list of str or str, optional):
            Atom selection strings specifying the helical regions to analyze.
            Defaults to ['name BB and resnum 500-520', 'name BB and resnum 525-556'].

        verbose (bool, optional):
            If True, prints the keys of the results dictionary. Defaults to True.

    Returns:
        dict:
            Dictionary containing results from the HELANAL analysis.
    """

    if isinstance(universe, md.core.trajectory.Trajectory):
        mdtraj_trajectory = universe
    
    elif isinstance(universe, mda.core.universe.Universe):
        mdtraj_trajectory = convert_mdanalysis_to_mdtraj(universe, topology)
    else:
        print(f'type not supported: {type(universe)}\n'
              f'needs type trajectory:md.core.trajectory.Trajectory or mda.core.universe.Universe')
    trajectory_no_jump = make_whole_mdtraj(mdtraj_trajectory)
    mdanalysis_universe = convert_mdtraj_to_mdanalysis(trajectory_no_jump, topology)
    helanal = hel.HELANAL(mdanalysis_universe, select=helix_selection)
    helanal.run()
    if verbose:
        print(dict(helanal.results).keys())
    return dict(helanal.results)

def test_network(network, descriptors, reference):
    """
    Perform one epoch of training using DataLoader.

    Parameters:
        network: initialized network (torch.nn.Module)
        epochs: number of epochs (int)
        optimizer: optimizer (torch.optim)
        dataloader: DataLoader for the training dataset
        device: device for the network (torch.device)
        dtype: data type for the network (torch.dtype)
        stop: stopping criterion, default: 30 (float)
        save_to: path to save the network, if argument is not provided, the state dicts will not be saved (str)
    
    Returns:
        losses: losses of the training (list)
    """       
    # TEST
    validation_loss = []
    estimate = expit(aimmd.evaluate(network, np.squeeze(np.array(descriptors))))
    for i in np.arange(len(estimate)):
        error = np.abs(reference[i]-estimate[i])/(
            reference[i] if reference[i] < .5 else 1 - reference[i])
        error = np.abs(reference[i]-estimate[i])
        validation_loss.append(error)
    return np.mean(validation_loss)