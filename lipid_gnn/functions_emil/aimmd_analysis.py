import os
import sys
import glob
import pathlib
import numpy as np
from pathlib import Path
import mdtraj as md
import inspect
from matplotlib.collections import LineCollection
import torch
import itertools

import utils as aimmd
from pathensemble import *

from functions import pkl_load, pkl_save
from tqdm import tqdm

def process_descriptors(descriptors):
    return descriptors

def get_validation_dataset_2(directory:(str, Path),
                           descriptor_fuction,
                           membrane_compositions:list= None,
                           verbose=True):
    
    """
    Load and assemble a validation dataset from pickled committor-check results.

    This function scans a given directory for `committor_check/*.pkl` files, loads
    their associated reference data, reconstructs the corresponding trajectories,
    and applies a user-provided descriptor function to each trajectory. The results
    are grouped by membrane composition (inferred from the directory structure) and
    returned as a dictionary mapping composition names to descriptor arrays and
    reference values.

    Parameters
    ----------
    directory : str or pathlib.Path
        Root directory to search. Expected to contain subdirectories with a
        `committor_check/` folder holding `.pkl` files and a `ref_frames/`
        folder containing `.gro` frame files.
    descriptor_fuction : callable
        A function that takes an MDAnalysis trajectory object and returns a
        descriptor array for that trajectory.
    membrane_compositions : list, optional
        If provided, only datasets whose composition name matches an element
        of this list will be included.
    verbose : bool, default True
        If True, displays a progress bar via `tqdm` while scanning files.

    Returns
    -------
    dict
        A dictionary mapping membrane composition names (str) to a list of:
        `[descriptors_array, reference_values]`, where
        - `descriptors_array` is a NumPy array of shape (N, ...) containing
          descriptors for each trajectory.
        - `reference_values` is a list of reference committor values extracted
          from the `.pkl` files.

    Notes
    -----
    - Trajectories whose `.gro` frame files cannot be loaded are skipped.
    - Descriptor outputs are squeezed and stored using `np.asanyarray`.
    """

    
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
                trajectory = mda.Universe(pkl_path.parent / 'ref_frames' / f'{key}.gro').trajectory
            except:
                continue
            reference.append(ref_results[key][3])
            descriptors.append(np.squeeze(descriptor_fuction(trajectory)))
        validation_dict[name] = [np.asanyarray(descriptors), reference]
        
    return validation_dict

def old_fit_function(network, pathensemble, validation, initial_path=None, verbose=False,
                     epochs=1000, lr=1e-3, save_to='best.h5', batch_size=4096):
    
    """
    Fit the neural network model according to the AIMMD simulation
    results, minimizing a log binomial loss.
   
    Input: network, our "Network" object.
           pathensemble, "PathEnsemble" object containing all the
                         AIMMD simulations up to this moment
                         (both shooting and free trajectories).
           validation, to check performance during the training
           initial_path, "PathEnsemble" object containing only the
                         initial path of our AIMMD run.
           verbose, if True, show a progress bar.
           epochs, number of training iterations
           lr, learning rate 
           save_to, path where the model is saved to
           batch_size, training batch size of the data
   
    Output: losses, list of losses over the training epochs.
            scales, list of maximum absolute network output over
                    the training epochs.
            values, array of neural network values of the training
                    set *before* the training.
            weights, array of weights associated to each training
                     set element.

    Originally compesed just by shooting points and two-way shooting
    results, the training set is now augmented by free simulation data
    in order to improve the training performance close to the states.
    """
    smoothening_weight = 100
    regularization_weight = 1e-4
    stop = 50.
    

    t0 = time.time()
    losses, scales, values, weights = [], [], [], []

    shots, equilibriumA, equilibriumB = aimmd.scorporate_pathensembles(
        pathensemble)
    equilibrium = equilibriumA + equilibriumB
    if not np.sum(equilibrium.frame_states == 'A'):
        temp = initial_path[:].unsplit()
        equilibriumA = temp.crop(frame_indices=temp.frame_states =='A')
    if not np.sum(equilibrium.frame_states == 'B'):
        temp = initial_path[:].unsplit()
        equilibriumB = temp.crop(frame_indices=temp.frame_states =='B')
    equilibrium = equilibriumA + equilibriumB

    dim = len(equilibrium.frame_descriptors[0])
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # collect all descriptors and results
    if len(shots):
        shooting_descriptors = shots.shooting_descriptors.reshape(
            (-1, dim)).astype(np.float64)
        shooting_results = shots.shooting_results
        shooting_weights = shots.are_accepted.astype(float)
        shooting_values = shots.shooting_values
    else:
        shooting_descriptors = np.zeros((0, dim), dtype=np.float64)
        shooting_results = np.zeros((0, 2))
        shooting_weights = np.zeros(0)
        shooting_values = np.zeros(0)
    k = (equilibrium.internal_states == 'R') * equilibrium.are_accepted
    kA = k * (equilibrium.initial_states == 'A')
    kB = k * (equilibrium.initial_states == 'B')
    if not np.sum(kA):
        kA = ((equilibrium.internal_states == 'A') +
              (equilibrium.final_states == 'A'))
    if not np.sum(kB):
        kB = ((equilibrium.internal_states == 'B') +
              (equilibrium.final_states == 'B'))
    equili_A_descriptors = np.concatenate(equilibrium.descriptors(
        kA, internal=True), axis=0).astype(np.float64)
    equili_A_results = np.repeat([[2., 0.]],
        equilibrium.internal_lengths[kA].sum(), axis=0)
    equili_A_values = np.concatenate(
        equilibrium.values(kA, internal=True))
    nA = max(1, len(equili_A_results))
    equili_B_descriptors = np.concatenate(equilibrium.descriptors(
        kB, internal=True), axis=0).astype(np.float64)
    equili_B_results = np.repeat([[0., 2.]],
        equilibrium.internal_lengths[kB].sum(), axis=0)
    equili_B_values = np.concatenate(
        equilibrium.values(kB, internal=True))
    nB = max(1, len(equili_B_results))
    scaleA = 1.
    scaleB = 1.
    if np.sum(shooting_results):
        scaleA *= min(1., np.sum(shooting_results) / nA / 2)
        scaleB *= min(1., np.sum(shooting_results) / nB / 2)
    equili_A_weights = np.repeat(scaleA, nA)
    equili_B_weights = np.repeat(scaleB, nB)

    print(f'shooting size {len(shooting_weights)}')
    print(f'eq A size {len(equili_A_weights)}')
    print(f'eq B size {len(equili_B_weights)}')

    # put everything together
    results = np.concatenate([shooting_results,
                              equili_A_results,
                              equili_B_results], axis=0)
    descriptors = process_descriptors(
        np.concatenate([shooting_descriptors,
                                  equili_A_descriptors,
                                  equili_B_descriptors], axis=0))
    weights = np.concatenate([shooting_weights,
                              equili_A_weights,
                              equili_B_weights])
    values = np.concatenate([shooting_values,
                             equili_A_values,
                             equili_B_values])

    weights = np.nan_to_num(weights)
    weights /= np.sum(weights)

    # training loop
    losses = []
    scales = []
    min_loss = np.inf
    validation_losses = []

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    for epoch in pbar:
        pbar.set_description('Training')

        # sample batch
        indices = np.random.choice(len(weights), batch_size, p=weights)
        d = torch.tensor(descriptors[indices], dtype=dtype, device=device)
        d.requires_grad = True
        r = torch.tensor(results[indices], dtype=dtype, device=device)

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
        losses.append(float(loss) / batch_size)

        # report scales
        q = network(d)
        scales.append(max(float(torch.max(q)), -float(torch.min(q))))
        Range = float(torch.min(q)), float(torch.max(q))

        # TEST
        network.eval()
        error_list = []
        conformations, reference = validation
        estimate = expit(aimmd.evaluate(network, np.squeeze(np.asanyarray(conformations)), batch_size=batch_size))
        for j in np.arange(len(estimate)):
            error = np.abs(reference[j]-estimate[j])#/(
#                     reference[j] if reference[j] < .5 else 1 - reference[j])
            error_list.append(error)
        coef = np.polyfit(reference, estimate, 1)
        correction = np.abs(1-coef[0]) + np.abs(0-coef[1])/2

        # get average of the batches
        validation_losses.append(np.mean(error_list) + correction/2)

        # save the model if the test loss is the lowest
        if validation_losses[-1] < min_loss:
            min_loss = validation_losses[-1]
            if save_to:
                # update the best model
                pkl_save(save_to, network.state_dict())



        pbar.set_postfix({'Epoch':epoch,
                          'Train loss':"%.3g" %losses[-1],
                          'validation loss':np.round(validation_losses[-1], 3),
                          'best validation loss': np.round(np.min(validation_losses), 3),
                          'scales':np.round(scales[-1], 3)}) 


    return losses, scales, values, weights

def new_fit_function(network, pathensemble, validation=None, initial_path=None, verbose=False,
        keys=None, save_memory=False, epochs=1000, lr=1e-3, save_to='best.h5', batch_size=4096,
                     mask=None):

    """
    Fit the neural network model according to the AIMMD simulation
    results, minimizing a log binomial loss.
   
    Input: network, our "Network" object.
           pathensemble, "PathEnsemble" object containing all the
                         AIMMD simulations up to this moment
                         (both shooting and free trajectories).
           validation, to check performance during the training
           initial_path, "PathEnsemble" object containing only the
                         initial path of our AIMMD run.
           verbose, if True, show a progress bar.
           keys, if not None, train only on the pathensemble paths
                 indexed by "keys." Good for bootstrapping or block-
                 averaging.
           save_memory, if True, execute "process_descriptors" on
                        every training batch, to avoid running out
                        of memory. This makes sense only if the whole
                        training set is so big it does not fit into
                        your computer RAM. (Hardly the case of
                        retinal's.)
            epochs, number of training iterations
            lr, learning rate 
            save_to, path where the model is saved to
            batch_size, training batch size of the data
            mask, excluding some artifacts from the training data
   
    Output: losses, list of losses over the training epochs.
            scales, list of maximum absolute network output over
                    the training epochs.
            values, array of neural network values of the training
                    set *before* the training.
            weights, array of weights associated to each training
                     set element.

    Originally compesed just by shooting points and two-way shooting
    results, the training set is now augmented by free simulation data
    in order to improve the training performance close to the states.
    """
   
    smoothening_weight = 1e-3
    regularization_weight = 1e-3
    stop = 30.
    nbins = 10
   
    pathensemble.update_values()  # with previous model

    t0 = time.time()
    losses, scales, selection_probabilities, results = [], [], [], []

    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

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
    
    if type(mask) != type(None):
        descriptor_mask = np.argmin(descriptors[:, mask], axis=1) > 44
        descriptors = descriptors[descriptor_mask, :]
        values = values[descriptor_mask]
        results = results[descriptor_mask, :]
        selection_probabilities = selection_probabilities[descriptor_mask]
        selection_probabilities /= np.sum(selection_probabilities)
        
    if not save_memory:  # all together now
        descriptors = process_descriptors(descriptors)

    """
    Training loop.
    """

    losses = []
    scales = []

    # in case of problems, restore this
    min_loss = np.inf
#         state_dict = copy.deepcopy(network.state_dict())
    validation_losses = []

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    for epoch in pbar:
        pbar.set_description('Training')

        for param_group in optimizer.param_groups:
            # slowly increase lr
            param_group['lr'] = lr * min(1, (epoch + 1) / (epochs / 20))

        # sample batch
        indices = np.random.choice(len(selection_probabilities),
                                   batch_size, p=selection_probabilities)
        if save_memory:  # separately to save memory
            d = process_descriptors(descriptors[indices])
        else:
            d = descriptors[indices]
        d = torch.tensor(d, dtype=dtype, device=device)
        d.requires_grad = True
        r = torch.tensor(results[indices], dtype=dtype, device=device)

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
        if validation:
            conformations, reference = validation
            estimate = expit(aimmd.evaluate(network, np.squeeze(np.asanyarray(conformations)), batch_size=batch_size))
            for j in np.arange(len(estimate)):
                error = np.abs(reference[j]-estimate[j])/(
                        reference[j] if reference[j] < .5 else 1 - reference[j])
                error_list.append(error)
            coef = np.polyfit(reference, estimate, 1)
            correction = np.abs(1-coef[0]) + np.abs(0-coef[1])/2

            # get average of the batches
            validation_losses.append(np.mean(error_list) + correction)

            # save the model if the test loss is the lowest
            if validation_losses[-1] < min_loss:
                min_loss = validation_losses[-1]
                if save_to:
                    # update the best model
                    pkl_save(save_to, [network.input_parameters, network.state_dict()])
        else:
            validation_losses.append(losses[-1])
            if validation_losses[-1] < min_loss:
                min_loss = validation_losses[-1]
                if save_to:
                    # update the best model
                    pkl_save(save_to, [network.input_parameters, network.state_dict()])



        pbar.set_postfix({'Epoch':epoch,
                          'Train loss':"%.3g" %losses[-1],
                          'validation loss':np.round(validation_losses[-1], 3),
                          'best validation loss': np.round(np.min(validation_losses), 3),
                          'scales':np.round(scales[-1], 3)}) 


    return losses, scales, selection_probabilities, results



def get_paths(search_string:str, key_index = -1, verbose=True):

    """
    Collect filesystem paths matching a search pattern and organize them in a dictionary.

    This function searches for files matching a given `search_string` pattern using
    `glob.glob`. Each matched path is stored in a dictionary with a key derived from
    one of the path's parts (default is the last part). Optionally, it prints the
    sorted keys of the resulting dictionary.

    Parameters
    ----------
    search_string : str
        Glob-style pattern to search for files (e.g., `'data/**/*.txt'`).
    key_index : int, default -1
        Index of the part of the path to use as the dictionary key. Negative
        indices count from the end.
    verbose : bool, default True
        If True, prints the sorted keys of the resulting dictionary.

    Returns
    -------
    dict
        A dictionary mapping selected path parts (keys) to `pathlib.Path` objects
        (values) representing the full matched paths.

    Notes
    -----
    - If multiple paths have the same key, the last one found will overwrite previous
      entries.
    - Sorting of keys when printing is based on the last 7 characters of each key.
    """

    path_list = {}
    for path in glob.glob(search_string):
        path_list[Path(path).parts[key_index]] = Path(path)
    if verbose:
        print('keys of dictionary:\n')
        for key in sorted(path_list.keys(), key=lambda arr: arr[-7:]):
            print(key)
    return path_list


def get_total_simulation_time(pathensemble, completion_time, dt):
    
    """
    Compute the total simulation time for trajectories up to a given completion time.

    This function calculates the cumulative simulation time of a set of trajectories
    represented by a pathensemble object, considering only those trajectories whose
    completion times are less than or equal to the specified `completion_time`.

    Parameters
    ----------
    pathensemble : object
        An object representing a collection of trajectories. Used attributes:
        - `completion_times` : array-like, the completion time of each trajectory.
        - `lengths` : array-like, the length (number of steps) of each trajectory.
    completion_time : float
        Maximum completion time to consider when summing trajectories.
    dt : float
        Time increment per step (timestep) of the trajectories.

    Returns
    -------
    float
        Total simulation time of all selected trajectories, in the same time units
        as `dt`.

    Notes
    -----
    - The subtraction of 2 from trajectory lengths ensures that very short trajectories
      do not contribute negative time; any negative values are clipped to zero.
    """

    
    keepers = (pathensemble.completion_times <= completion_time)
    return np.sum(np.maximum(pathensemble.lengths[keepers] - 2, 0)) * dt

def total_simulation_time(pathensemble,
                                   dt:float):
    
    """
    Compute cumulative simulation time as a function of increasing completion times.

    This function iterates over all unique completion times in the given pathensemble,
    computing the total simulation time up to each completion time using
    `get_total_simulation_time`. The results are returned as a NumPy array with two
    columns: completion times and the corresponding cumulative simulation times.

    Parameters
    ----------
    pathensemble : object
        An object representing a collection of trajectories. Used attributes:
        - `completion_times` : array-like, the completion time of each trajectory.
        - `lengths` : array-like, the length (number of steps) of each trajectory.
    dt : float
        Time increment per step (timestep) of the trajectories.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape `(N, 2)` where each row corresponds to:
        - Column 0: completion time `i`
        - Column 1: total simulation time accumulated up to completion time `i`.

    Notes
    -----
    - The function sorts completion times in ascending order before processing.
    - Uses `get_total_simulation_time` to calculate cumulative time at each step.
    - Resulting array is suitable for plotting cumulative simulation time vs. completion time.
    """

    result = []
    for i in sorted(pathensemble.completion_times):
        result.append([i, get_total_simulation_time(pathensemble, i, dt)])
    return np.asanyarray(result)

def dRMSD(trajectory, Connections_ref):

    """
    Compute the distance root-mean-square deviation (dRMSD) for a trajectory.

    This function calculates the dRMSD of a molecular trajectory with respect to a
    reference set of interatomic distances. It compares the pairwise distances
    specified in `Connections_ref` between the current trajectory frames and the
    reference distances, returning the RMS deviation for each set of distances.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        The molecular dynamics trajectory to analyze.
    Connections_ref : numpy.ndarray
        Array of shape `(M, 3)` defining reference distances. Each row should
        contain `[atom_index1, atom_index2, reference_distance]`.

    Returns
    -------
    numpy.ndarray
        1D array of dRMSD values for each frame in the trajectory.

    Notes
    -----
    - RMSD is computed as:

      ```
      sqrt( sum((d - d0)**2) / (n - 1) )
      ```
      where `d` are the distances in the current frame, `d0` are the reference
      distances, and `n` is the number of distances.
    """

    
    d = md.compute_distances(trajectory,
                             Connections_ref[:, :2].astype(int))
    d0 = Connections_ref[:, 2]
    n = len(Connections_ref)
    return np.sqrt(np.sum((d - d0) ** 2, axis=1) / (n-1))

        
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
    
    if isinstance(trajectory, list):
        atom_number = len(trajectory[0].universes[0].atoms)
    else:
        atom_number = len(trajectory.universes[0].atoms)
    position_array = []
    unitcell_vectors_array = []
    time_array = []
    new_traj = md.load(topology)
    for frame in trajectory:
        try:
            if isinstance(trajectory, list):
                frame = frame[0]
        except:
            continue
        positions = frame.positions
        position_array.append(positions.reshape((-1, *positions.shape)) / 10.)
        unitcell_vectors_array.append(frame.triclinic_dimensions.reshape((-1, 3, 3)) / 10.)
        time_array.append(frame.time)
    new_traj.xyz = np.vstack(position_array)
    new_traj.unitcell_vectors = np.vstack(unitcell_vectors_array)
    new_traj.time = np.asanyarray(time_array)
    return new_traj

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

def precalculate_CV(Pathensemble,
                    function, 
                    save_dir:str,
                    name:str='New_CV',
                    function_parameter=None,
                    skip=True):
    
    """
    Precompute collective variable (CV) values for a set of trajectories and save them.

    This function iterates over all trajectories in a pathensemble, applies a user-provided
    function to calculate collective variables (CVs), and stores the results in an HDF5
    file. If the CV values for a given trajectory already exist and `skip=True`, computation
    is skipped. Any errors encountered during trajectory loading or function execution are
    logged in the dictionary under the 'error' key.

    Parameters
    ----------
    Pathensemble : object
        An object representing a collection of trajectories. Used attributes:
        - `frame_trajectory_indices`
        - `frame_trajectory_positions`
        - `trajectory_files`
        - `trajectory_directories`
        - `trajectory_topologies`
    function : callable
        A function to compute collective variables on an MDTraj trajectory. Should accept
        a trajectory and optional parameters.
    save_dir : str
        Directory where the HDF5 file will be saved.
    name : str, default 'New_CV'
        Name of the HDF5 file (without extension) to save the computed CVs.
    function_parameter : any, optional
        Additional parameters to pass to `function`.
    skip : bool, default True
        If True, skip trajectories that already have computed CV values in the file.

    Returns
    -------
    dict
        Dictionary containing the precomputed CVs. Structure:
        - Keys: function names or individual CV names if `function` returns a dictionary.
        - Values: dictionaries mapping trajectory keys to computed CV arrays.
        - 'error': string concatenating any errors encountered during computation.

    Notes
    -----
    - Each trajectory is loaded using MDTraj (`md.load`) with its associated topology.
    - The HDF5 file is updated incrementally after each trajectory to prevent data loss.
    - If the CV function returns a NumPy array, it is stored under the function name.
      If it returns a dictionary, each key is stored separately.
    - Trajectories are identified by the last two path components: `'/'.join(path.parts[-2:])`.
    """

    
    # get frame_trajectory_indices and positions
    frame_trajectory_indices = Pathensemble.frame_trajectory_indices.ravel()
    frame_trajectory_positions = Pathensemble.frame_trajectory_positions.ravel()
    
    # get trajectories
    indices = np.unique(frame_trajectory_indices)

    # get trajectories
    files = np.array(Pathensemble.trajectory_files)[indices]
    directories = np.array(Pathensemble.trajectory_directories)[indices]
    topologies = np.array(Pathensemble.trajectory_topologies)[indices]
    
    # try loading same h5 file
    CV_dictionary = pkl_load(f'{str(save_dir)}/{name}.h5')
    if not CV_dictionary:
        CV_dictionary = {}
    
    # prime error
    if 'error' not in list(CV_dictionary.keys()):
        CV_dictionary['error'] = ''
        
    # go to every file in pathensamble and save values from functions in dict
    for file, directory, topology in tqdm(zip(files, directories,
                                              topologies),
                                          total=len(topologies)):
        
        # get key and check if it is already in dict
        key = '/'.join(Path(f'{directory}/{file}').parts[-2:])
        if len(CV_dictionary.keys()) > 1 and skip:
            check = [key in CV_dictionary[CV_dictionary_key].keys() for CV_dictionary_key in CV_dictionary 
                     if CV_dictionary_key != 'error']
            if all(check):
                continue
            
        # try loading in file and write down the error if it is not working 
        try:
            trajectory = md.load(f'{directory}/{file}', top=f'{directory}/{topology}')
            function_output = function(trajectory, function_parameter)
            
        except Exception as error:
            CV_dictionary['error'] += f'{error}\n'
            continue
            
        if isinstance(function_output, np.ndarray):
            if function.__name__ not in list(CV_dictionary.keys()):
                CV_dictionary[function.__name__] = {}
            CV_dictionary[function.__name__][key] = function_output
            
        if isinstance(function_output, dict):
            for function_key in function_output:
                if function_key not in list(CV_dictionary.keys()):
                    CV_dictionary[function_key] = {}
                CV_dictionary[function_key][key] = function_output[function_key]
            
        pkl_save(f'{str(save_dir)}/{name}.h5', CV_dictionary)
    return CV_dictionary
        
def get_simulation_timestep(path:(str, pathlib.Path)):
    
    """
    Extract simulation timestep and output frequencies from a GROMACS-style input file.

    This function reads a simulation input file (e.g., `.mdp` file) and extracts the
    timestep (`dt`) and output frequencies (`nstxout`, `nstvout`, etc.). It also
    calculates the effective total save interval in picoseconds, nanoseconds, and
    microseconds.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the simulation input file (plain text) to parse.

    Returns
    -------
    dict
        Dictionary containing:
        - Keys corresponding to parameters in the input file (e.g., 'dt', 'nstxout', 'nstvout') 
          with their values as floats.
        - 'ps' : effective total save interval in picoseconds.
        - 'ns' : effective total save interval in nanoseconds.
        - 'µs' : effective total save interval in microseconds.

    Notes
    -----
    - Assumes the input file uses GROMACS-style `parameter = value` syntax.
    - Only the first line containing `dt` is considered for the timestep.
    - The main save interval is calculated as the maximum of all `nstxout`/`nstvout` values
      multiplied by the timestep.
    """

    
    lines = open(path, 'r').read().split('\n')
    
    result_dict = {}
    time_step  = [[line.split('=')[0].split(' ')[0], float(line.split('=')[-1])] for line in lines 
                   if not line == '' 
                   if 'dt' in line.split()[0]][0]
    
    save_time  = [[line.split('=')[0].split(' ')[0], float(line.split('=')[-1])] for line in lines 
                   if not line == '' 
                   if 'nstxout' in line.split()[0] or
                   'nstvout' in line.split()[0]]
    
    main_save_time = np.max([float(time[-1]) for time in save_time])
    
    result_dict[time_step[0]] = time_step[-1]
    for i in save_time:
        result_dict[i[0]] = i[-1]
    
    result_dict['ps'] = main_save_time*time_step[-1]
    result_dict['ns'] = main_save_time*time_step[-1]*1e-3
    result_dict['µs'] = main_save_time*time_step[-1]*1e-6
    
    return result_dict

def _project_preclaculated_CV(pathensemble,
                             path_to_precalculated_CV:str,
                             functions:str = None,
                              concatenate = True,
                              frame_indices=None):
    
    """
    Project a Pathensemble onto precomputed collective variables (CVs) with optional weighting.

    This function loads precomputed CVs from a serialized file and maps the frames
    of a `Pathensemble` to their corresponding CV values. It also applies optional
    filters to exclude frames based on specified criteria (e.g., artifacts identified
    via spherical coordinates). The function can return concatenated results or
    maintain separate arrays per ensemble.

    Parameters
    ----------
    pathensemble : iterable
        Collection of ensemble objects, each expected to provide:
        - `trajectory_positions(internal=True)`
        - `trajectory_directories`
        - `trajectory_files`
        - `weights`
    path_to_precalculated_CV : str
        Path to a file containing precomputed CV values (serialized dictionary).
    functions : str or None, optional
        List of CV functions to project onto. Defaults to all functions in the file
        except 'error'.
    concatenate : bool, default True
        If True, concatenate all CV arrays and weights across ensembles; if False,
        maintain separate arrays per ensemble.
    frame_indices : array-like or None, optional
        Specific frame indices to project. If None, all frames are used.

    Returns
    -------
    tuple of dict
        - results : dict
            Mapping of CV names to arrays of projected values (concatenated or per-ensemble).
        - weights : dict
            Mapping of CV names to arrays of corresponding weights.

    Notes
    -----
    - Frames are filtered using `eigenvector_lower_helix_monA` and `eigenvector_lower_helix_monB`,
      excluding frames where the polar angle of the transmembrane helix exceeds 70° in either monomer.
    - Trajectories missing CV values or with insufficient frames are skipped.
    - Uses `cartesian_to_spherical` internally to identify artifacts.
    - Assumes CVs in `path_to_precalculated_CV` are stored per trajectory as NumPy arrays.
    """

    
    original_dict = pkl_load(path_to_precalculated_CV)
    function_list = [function for function in list(original_dict.keys()) if function != 'error']
        
    values = {}
    mask = {}
    weights = {} 
    for ensemble in pathensemble:
        frame_positions = ensemble.trajectory_positions(internal=True)[0]
        positions_per_file = np.split(frame_positions, np.where(frame_positions == 0)[0])
        positions_per_file = [i for i in positions_per_file if not i.size == 0]
        for function in function_list:
            if function not in list(values.keys()) or function not in list(values.keys()):
                mask[function] = []
                values[function] = []
                weights[function] = []
            skip = False
            CV = []
            keep = []
            for number in np.arange(len(positions_per_file)):
                path = Path(f'{ensemble.trajectory_directories[number]}/{ensemble.trajectory_files[number]}')
                file = '/'.join(path.parts[-2:])
                if file not in list(original_dict[function].keys()):
                    skip =True
                    break
                if len(original_dict[function][file])-1 < np.max(positions_per_file[number]):
                    skip =True
                    break
                CV.append(original_dict[function][file][positions_per_file[number]])
            
                # during 
                monA = original_dict['eigenvector_lower_helix_monA'][file][positions_per_file[number]]
                monB = original_dict['eigenvector_lower_helix_monB'][file][positions_per_file[number]]
                is_not_artifact = (cartesian_to_spherical(monA)[2] <70) * (cartesian_to_spherical(monB)[2] <70)
                keep.append(is_not_artifact)
            if skip:
                continue
            
            
            
            keep = np.concatenate(keep)
            CV = np.concatenate(CV)[keep]
            values[function].append(CV)
            weights[function].append(np.repeat(ensemble.weights[0], len(CV)))
    
    if concatenate:
        results = {}
        weights_result = {}
        for key in values:
            if values[key] == []:
                results[key] = np.array([])
                weights_result[key] = np.array([])
            else:
                results[key] = np.concatenate(values[key])
                weights_result[key] = np.concatenate(weights[key])
    else:
        results = values
        weights_result = weights

    return results, weights


def project(ensemble, bins=[-np.inf, +np.inf],
                key=None, use_precalculated_CV=None,
            bootstrapping_iterations:int=0, 
           values_weights_only=False):

    """
    Project an ensemble or collection of ensembles onto precalculated collective
    variables (CVs) and compute multidimensional histograms with optional
    bootstrapping.

    Parameters
    ----------
    ensemble : Pathensemble or aimmd.PathEnsemblesCollection
        The object to project.
    bins : array-like, default [-inf, +inf]
        Histogram bin edges. A 1D array is reshaped to `(1, -1)`.
    key : int, slice, array-like or None
        Selector used when projecting a PathEnsemblesCollection.
    use_precalculated_CV : tuple
        Tuple `(cv_data_path, cv_list)` indicating:
        - the source of precalculated CV values,
        - the list of CV names to read.
    bootstrapping_iterations : int, default 0
        Number of bootstrap resampling iterations.
    values_weights_only : bool, default False
        If True, return raw values and weights without histogramming.

    Returns
    -------
    values, weights : lists of ndarrays
        Returned only if `values_weights_only=True`.
    results_hist : ndarray
        Multidimensional histogram of the CV values.
    bootstrapping_results : ndarray
        Histogram results from each bootstrap iteration.

    Notes
    -----
    - When projecting a PathEnsemblesCollection, the function groups contributions
      based on parent pathensembles.
    - Only pathensembles with ≥4 frames are used.
    - Bootstrapping samples CV entries with replacement before histogramming.
    - The final histogram is returned with `.T` to match typical plotting
      conventions.
    """

    bins = np.array(bins)
    if len(bins.shape) == 1:
        bins = bins.reshape(1, -1)

    if type(ensemble) == aimmd.PathEnsemblesCollection:

        # select the right key
        key = np.arange(len(ensemble))[key].ravel()
        pathensemble_indices = ensemble.pathensemble_indices[key]
        path_indices = ensemble.path_indices[key]

        # a key for each pathensemble in Collection
        values = []
        weights = []
        for i in np.unique(pathensemble_indices):
            pathensemble = ensemble.pathensembles[i]
            if pathensemble.nframes < 4:
                continue
            values_temp, weights_temp = _project(pathensemble,
                                            use_precalculated_CV[0],
                                            functions=use_precalculated_CV[1],
                                            frame_indices=None,
                                            concatenate=False)
            values.append(values_temp[use_precalculated_CV[1][0]])
            weights.append(weights_temp[use_precalculated_CV[1][0]])


    else:
        values, weights = _project(ensemble,
                                    use_precalculated_CV[0],
                                    functions=use_precalculated_CV[1],
                                    frame_indices=None,
                                    concatenate=False)

    values = list(itertools.chain.from_iterable(values))
    weights = list(itertools.chain.from_iterable(weights))
    
    if values_weights_only:
        return values, weights

    # bootstrapping
    bootstrapping_results = []
    for i in np.arange(bootstrapping_iterations):
        random_choice = np.random.choice(len(values), size=len(values))
        bootstrap_weights = [weights[i] for i in random_choice]
        bootstrap_values = [values[i] for i in random_choice]
        bootstrapping_results.append(np.histogramdd(np.concatenate(bootstrap_values),
                                                    bins,
                                                    density=True,
                                                    weights=np.concatenate(bootstrap_weights))[0].T)

    values = np.concatenate(values)
    weights = np.concatenate(weights)
    results_hist = np.histogramdd(
        values, bins, density=True, weights=weights)[0].T
    

    return results_hist, np.asanyarray(bootstrapping_results)


'''def _project(pathensemble, bins=[-np.inf, +np.inf],
            key=None, f=None, frames=False,
            weights=None, vmin=None, vmax=None,
            backward=True, forward=True, use_precalculated_CV=None,
            bootstrapping_iterations:int=0):
    """
    Parameters
    ----------
    bins: array-like of floats
          borders of the bins
          the dimension is guessed from here
    f: callable function
       run through all the paths in the path ensemble
       if None: take pathensemble.values
    frames: if True: f takes mdanalysis frames as input, otw descr.
    weights: np.array of weights of the path ensemble paths;
             if None: use standard weights
    vmin, vmax: project only points with RC values between vmin and vmax

    Returns
    -------
    distribution: array-like of size (len(bins) - 1)
                  counts the population in each bin (sum of the weights)
    """

    # process bins
    mask = None
    bins = np.array(bins)
    if len(bins.shape) == 1:
        bins = bins.reshape(1, -1)

    # process weights
    if weights is None:
        weights = pathensemble.weights[key].ravel()

    # override frames option
    if not pathensemble.frame_descriptors.shape[1]:
        frames = True

    # extract frames and weights
    frame_weights = np.zeros(pathensemble.nframes)
    for frame_indices, weight, is_accepted in zip(
        pathensemble.frame_indices(key, internal=True,
                           backward=backward, forward=forward),
        weights, pathensemble.are_accepted[key].ravel()):
        frame_weights[frame_indices] += weight * is_accepted
    if vmin is not None:
        frame_weights[pathensemble.frame_values < vmin] = 0.
    if vmax is not None:
        frame_weights[pathensemble.frame_values >= vmax] = 0.
    frame_indices = np.where(frame_weights > 0)[0]
    weights = frame_weights[frame_indices]  # now weights of frames

    # get values
    if use_precalculated_CV != None:
        values, mask, weights = get_precalculated_CV(pathensemble,
                                            use_precalculated_CV[0],
                                            functions=use_precalculated_CV[1],
                                            frame_indices=frame_indices,
                                            concatenate=False,
                                                    return_weights=True)

        
        if values:
            values = values[use_precalculated_CV[1][0]]
            weights = weights[use_precalculated_CV[1][0]]
        else:
            values = np.array([])
            weights = np.array([])
    elif f is None:
        values = pathensemble.frame_values[frame_indices]
    elif not frames:
        values = f(pathensemble.frame_descriptors[frame_indices])
    else:
        values = f(pathensemble.frames(frame_indices))
    
    return values, weights'''

def time_crop_pathensemble(pathensemble,
                      total_simulation_time:float = None,
                      times:np.array = None,
                      dt:float=1.0):

    """
    Crop a pathensemble in time by removing all trajectories and equilibrium segments
    that extend beyond a specified cumulative simulation time.

    Parameters
    ----------
    pathensemble : aimmd.Pathensemble or aimmd.PathEnsemblesCollection
        The ensemble to crop.
    total_simulation_time : float, optional
        Target total simulation time. The function finds the closest available
        completion time and crops to that point. If None, no cropping is applied.
    times : np.ndarray, optional
        Precomputed array from `total_simulation_time(pathensemble, dt)` giving
        cumulative simulation time as a function of completion time. If None, it is
        recomputed.
    dt : float, default 1.0
        Time step used for computing cumulative simulation time when `times` is not
        provided.

    Returns
    -------
    aimmd.Pathensemble
        A new pathensemble containing only:
        - shots with completion times below the chosen cutoff,
        - equilibriumA and equilibriumB segments cropped to the cutoff time.

    Notes
    -----
    - Cropping is performed by identifying the completion time whose accumulated
      simulation time most closely matches `total_simulation_time`.
    - If `total_simulation_time` is None, the input ensemble is returned unchanged
      (except for restructuring via `scorporate_pathensembles`).
    - The returned object is reconstructed by concatenating the cropped components.
    """


    shots, equilibriumA, equilibriumB = aimmd.scorporate_pathensembles(pathensemble)
    if times is None:
        times = total_simulation_time(pathensemble, dt)
    if total_simulation_time is None:
        keepers = None
        completion_time = np.inf
    else:
        completion_time = times[np.argmin(np.abs(times[:, 1] - total_simulation_time)), 0]
        keepers = shots.completion_times < completion_time
        
    shots = shots[keepers]
    equilibriumA = equilibriumA.crop(tmax=completion_time)
    equilibriumB = equilibriumB.crop(tmax=completion_time)
    return shots + equilibriumA + equilibriumB

def get_number_of_workers(pathensemble):
    
    """
    Count the number of worker log files associated with a pathensemble.

    This function scans each unique directory referenced by the pathensemble and
    counts files matching worker-style log patterns.

    Parameters
    ----------
    pathensemble : aimmd.Pathensemble
        Object whose `.directories` attribute lists simulation directories to scan.

    Returns
    -------
    int
        The total number of worker log files found across all directories.
    """


    workers = 0
    for i in np.unique(pathensemble.directories):
        workers += len(glob.glob(f'{i}/?.log'))
        workers += len(glob.glob(f'{i}/??.log'))
        workers += len(glob.glob(f'{i}/log'))
    return workers

def diagnostics(pathensemble, dt:float=1):
    
    """
    Generate a diagnostic summary for a pathensemble, reporting key statistics
    for shots, equilibriumA, and equilibriumB segments.

    This function separates the input pathensemble into its constituent parts,
    computes several descriptive metrics for each component, and returns a
    formatted text table.

    Parameters
    ----------
    pathensemble : aimmd.Pathensemble or compatible object
        The ensemble to analyze. Must be compatible with
        `aimmd.scorporate_pathensembles`.
    dt : float, default 1
        Time step used for converting frame counts into simulation time.

    Returns
    -------
    str
        A formatted multi-line string summarizing diagnostics, including:
            - number of workers
            - total number of frames
            - total simulation time
            - number of paths
            - average path length (frames and dt)
            - number of transition paths
            - transition-to-excursion ratio

    Notes
    -----
    - Worker counts are determined via `get_number_of_workers` and rely on
      log-file patterns inside each ensemble directory.
    """
    
    shots, equilibriumA, equilibriumB = aimmd.scorporate_pathensembles(pathensemble)
    
    def format_print(parameter:str, shots:str, equiA:str, equiB:str):
        return f'{parameter:35} | {shots:15} | {equiA:15} | {equiB:15} |\n'

    text = ''
    text += format_print('Parameter', 'Shots', 'Equilbrium A', 'Equilibrium B')
    text += '-------------------------------------------------------------------------------------------\n'
    workers_shots = get_number_of_workers(shots)
    workers_equilibriumA = get_number_of_workers(equilibriumA)
    workers_equilibriumB = get_number_of_workers(equilibriumB)
    if workers_shots == workers_equilibriumA == workers_equilibriumB == 0:
        text += format_print('Number of Workers',
                     '-',
                     '-',
                     '-')
    else:
        text += format_print('Number of Workers',
                     workers_shots,
                     workers_equilibriumA,
                     workers_equilibriumB)
    text += format_print('Total number of frames',
                 np.sum(shots.lengths -2),
                 np.sum(equilibriumA.lengths -2),
                 np.sum(equilibriumB.lengths -2))
    text += format_print('Total simulation time [dt]',
                 np.round(np.sum(shots.lengths -2)*dt, 2),
                 np.round(np.sum(equilibriumA.lengths -2)*dt, 2),
                 np.round(np.sum(equilibriumB.lengths -2)*dt, 2))
    text += format_print('Total number of paths',
                 len(shots),
                 len(equilibriumA),
                 len(equilibriumB))
    text += format_print('Average path length [frames]',
                 np.round(np.mean(shots.lengths -2), 2),
                 np.round(np.mean(equilibriumA.lengths -2), 2),
                 np.round(np.mean(equilibriumB.lengths -2), 2))
    text += format_print('Average path length [dt]',
                 np.round(np.mean(shots.lengths -2)*dt, 3),
                 np.round(np.mean(equilibriumA.lengths -2)*dt, 3),
                 np.round(np.mean(equilibriumB.lengths -2)*dt, 3))
    text += format_print('Number of transition paths',
                 np.sum(shots.are_transitions),
                 np.sum(equilibriumA.are_transitions),
                 np.sum(equilibriumB.are_transitions))
    text += format_print('Ratio of transition to excursions',
                 np.round(np.sum(shots.are_transitions)/len(shots), 5),
                 np.round(np.sum(equilibriumA.are_transitions)/len(equilibriumA), 5),
                 np.round(np.sum(equilibriumB.are_transitions)/len(equilibriumB), 5))
    return text
    
def load_pathensemble(states_function,
                      descriptors_function,
                      values_function,
                      directory,
                      update_descriptors=False,
                      verbose=True,
                      use_aimmd=False):
    
    """
    Load a PathEnsemblesCollection from a directory or, optionally, load a single
    pathensemble directly using AIMMD's built-in loader.

    This function provides two loading modes:
    1. **AIMMD mode** (`use_aimmd=True`): directly calls `aimmd.load_pathensemble`.
    2. **Manual mode**: loads multiple `.h5` files, constructs a
       `PathEnsemblesCollection`, and attaches state, descriptor, and value
       functions.

    Parameters
    ----------
    states_function : callable
        Function used by AIMMD to evaluate states for each frame.
    descriptors_function : callable
        Function computing descriptors for each trajectory.
    values_function : callable
        Function computing values associated with each frame.
    directory : str or Path
        Directory containing `.h5` trajectory files or AIMMD pathensemble data.
    update_descriptors : bool, default False
        If True, call `update_descriptors()` on the assembled pathensemble.
    verbose : bool, default True
        If True, print progress information.
    use_aimmd : bool, default False
        If True, bypass manual loading and call `aimmd.load_pathensemble`.

    Returns
    -------
    aimmd.Pathensemble or aimmd.PathEnsemblesCollection
        The loaded ensemble object.

    Notes
    -----
    - In manual mode, all `.h5` files in the directory are loaded, sorted
      numerically by filename stem.
    - The resulting collection is split into `shots`, `equilibriumA`,
      and `equilibriumB` via `aimmd.scorporate_pathensembles`.
    - Verbose mode prints summary information for each chain, including number
      of paths, transitions, and time since last update.
    - If `update_descriptors=True`, descriptors are recomputed after loading.
    """

    
    if use_aimmd:
        PATHENSEMBLE = aimmd.load_pathensemble(states_function,
                                       descriptors_function,
                                       values_function,
                                       directory=directory,
                                       verbose=verbose)
        return PATHENSEMBLE
    
    pathensembles = []
    for path in sorted(Path(directory).glob('*.h5'), key=lambda arr: int(arr.stem)):
        pathensembles.append(pkl_load(path))

    PATHENSEMBLE = aimmd.PathEnsemblesCollection(*pathensembles)
    PATHENSEMBLE.states_function = states_function
    PATHENSEMBLE.descriptors_function = descriptors_function
    PATHENSEMBLE.values_function = values_function
    
    shots, equilibriumA, equilibriumB = aimmd.scorporate_pathensembles(PATHENSEMBLE)
    if verbose:
        aimmd.write(f'shots: {shots}')
        t0 = time.time()
        for i, p in enumerate(shots.pathensembles):
            if len(p):
                t = t0 - p.completion_times[-1]
            else:
                t = 0
            nt = np.sum(p.are_transitions)
            aimmd.write(f'    chain {i}: {len(p)} paths, {nt} transitions, '
                  f'last updated {t:.0f} s ago')
        aimmd.write(f'equilibriumA: {equilibriumA}')
        aimmd.write(f'equilibriumB: {equilibriumB}')
        
    if update_descriptors:
        if verbose:
            write(f'Updating descriptors')
        pathensemble.update_descriptors(verbose=True)
    
    return PATHENSEMBLE


def get_precalculated_CV(pathensemble,
                         path_to_precalculated_CV:str,
                         frame_indices = None,
                         functions:str = None,
                         concatenate = True,
                         return_weights= False):
    
    """
    Retrieve precalculated collective variable (CV) values for a pathensemble,
    optionally filtered by specific functions and frame indices.

    Parameters
    ----------
    pathensemble : aimmd.Pathensemble
        The pathensemble for which to extract CV values.
    path_to_precalculated_CV : str
        Path to a pickled HDF5 file containing precalculated CV values.
    frame_indices : array-like, optional
        Indices of frames to select. If None, all frames are used.
    functions : list of str, optional
        List of CV function names to extract. If None, all functions except 'error'
        are returned.
    concatenate : bool, default True
        If True, concatenate values across trajectories into a single array per function.
    return_weights : bool, default False
        If True, return the associated weights along with CV values and mask.

    Returns
    -------
    results : dict
        Dictionary mapping function names to arrays of CV values.
    mask : dict
        Boolean mask arrays indicating which frames contributed to the results.
    weights : dict, optional
        Returned only if `return_weights=True`. Contains arrays of weights
        corresponding to each CV value.

    Notes
    -----
    - The function filters CVs according to `functions` and frame indices.
    - `mask` arrays allow tracking of which frames were successfully included.
    - Trajectory weights from the pathensemble are applied if `return_weights=True`.
    - Concatenation can be disabled to preserve per-trajectory structure.
    """

    
    original_dict = pkl_load(path_to_precalculated_CV)
    if functions:
        cut_dict = {'error':original_dict['error']}
        for function in functions:
            cut_dict[function] = original_dict[function]
        original_dict = cut_dict
        
    function_list = [function for function in list(original_dict.keys()) if function != 'error']
    
    # get frame_trajectory_indices and positions
    frame_trajectory_indices = pathensemble.frame_trajectory_indices[
        frame_indices].ravel()
    frame_trajectory_positions = pathensemble.frame_trajectory_positions[
        frame_indices].ravel()

    # get trajectories
    indices = np.unique(frame_trajectory_indices)
    files = np.array(pathensemble.trajectory_files)[indices]
    directories = np.array(pathensemble.trajectory_directories)[indices]
    
    key_list = []
    for file, directory in zip(files, directories):
        path = Path(f'{directory}/{file}')
        key_list.append('/'.join(path.parts[-2:]))
    new_frame_trajectory_indices = np.zeros(
        len(frame_trajectory_indices), dtype=int)
    for i, index in enumerate(indices):
        new_frame_trajectory_indices[frame_trajectory_indices == index] = i
    unique, unique_index, _ = np.unique(new_frame_trajectory_indices, return_counts=True, return_index=True)
    unique_index = np.append(unique_index, len(new_frame_trajectory_indices))
    values = {}
    mask = {}
    count = 0
    weights = {} 
    for path in unique:
        frames = frame_trajectory_positions[unique_index[path]:unique_index[path+1]]
        for function in function_list:
            if function not in list(values.keys()) or function not in list(values.keys()):
                mask[function] = []
                values[function] = []
                weights[function] = []
            if key_list[path] not in list(original_dict[function].keys()):
                mask[function].append(np.repeat(False, len(frames)))
                continue
            CV = original_dict[function][key_list[path]]
            length = len(CV)
            mask_values = np.asanyarray([i in frames for i in np.arange(length)])
            mask[function].append(np.asanyarray([i in np.arange(length) for i in frames]))
            values[function].append(CV[mask_values])
            weights[function].append(np.repeat(pathensemble.weights[path], len(mask_values)))
    
    if concatenate:
        results = {}
        weights_result = {}
        for key in values:
            if values[key] == []:
                results[key] = np.array([])
                weights_result[key] = np.array([])
            else:
                results[key] = np.concatenate(values[key])
                weights_result[key] = np.concatenate(weights[key])
            mask[key] = np.concatenate(mask[key])
    else:
        results = values
        weights_result = weights
    if return_weights:
        return results, mask, weights
    else:
        return results, mask



def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def compute_angle_in_array(arr, arr2):
    result = []
    for index in np.arange(len(arr)):
        result.append(angle_between(arr[index], arr2[index]))
    return np.asanyarray(result)

def appendSpherical_np(xyz):
    
    """
    Convert Cartesian coordinates to spherical coordinates and append them
    to the original array.

    Parameters
    ----------
    xyz : ndarray, shape (n_points, 3)
        Array of Cartesian coordinates (x, y, z).

    Returns
    -------
    ptsnew : ndarray, shape (n_points, 6)
        Array containing original Cartesian coordinates and corresponding
        spherical coordinates:
            - Column 0-2: original x, y, z
            - Column 3: radial distance r
            - Column 4: polar angle θ (degrees, from Z-axis down)
            - Column 5: azimuthal angle φ (degrees, from X-axis in XY-plane)

    Notes
    -----
    - Elevation/polar angle is measured from the positive Z-axis downward.
    - Azimuthal angle is measured counterclockwise from the positive X-axis.
    """

    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.degrees(np.arctan2(np.sqrt(xy), xyz[:,2])) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.degrees(np.arctan2(xyz[:,1], xyz[:,0]))
    return ptsnew

def correct_angle(sphere_angles):
    
    """
    Correct spherical angles to lie within the range [-180, 180] degrees and
    invert the polar angle.

    Parameters
    ----------
    sphere_angles : ndarray, shape (n_points, 2)
        Array of spherical angles, where column 0 is the polar/elevation angle
        and column 1 is the azimuthal angle.

    Returns
    -------
    sphere_angles_degree : ndarray, shape (n_points, 2)
        Corrected spherical angles:
            - Column 0: azimuthal angle wrapped to [-180, 180] degrees
            - Column 1: inverted polar/elevation angle

    Notes
    -----
    - This function ensures azimuthal angles outside [-180, 180] are wrapped.
    - Polar/elevation angles are inverted (multiplied by -1) to match the
      desired coordinate convention.
    """

    sphere_angles_degree = np.zeros(sphere_angles.shape)
    for i in np.arange(len(sphere_angles)):
        j = sphere_angles
        if j[i][1] > 180:
            sphere_angles_degree[i] = np.array([j[i][1]-360, -j[i][0]])
        elif j[i][1] < -180:
            sphere_angles_degree[i] = np.array([j[i][1]+360, -j[i][0]])
        else:
            sphere_angles_degree[i] = np.array([j[i][1], -j[i][0]])
    return sphere_angles_degree




def colored_line(angle, y, ax, dt=1, **lc_kwargs):
    
    """
    Create a colored line plot where the color varies along the line according
    to a trajectory.

    Parameters
    ----------
    angle : array-like
        X-coordinates of the line (e.g., angular values in degrees).
    y : array-like
        Y-coordinates of the line.
    ax : matplotlib.axes.Axes
        The matplotlib axes to which the line will be added.
    dt : float, default 1
        Time step or spacing used to generate the color array along the line.
    lc_kwargs : dict
        Additional keyword arguments passed to `matplotlib.collections.LineCollection`.
        Any provided "array" argument will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The line collection object added to the axes.

    Notes
    -----
    - The function handles discontinuities in `angle` when crossing ±180 degrees,
      adjusting line segments to avoid visual jumps.
    - Colors are assigned along the line using the sequence `c = linspace(0, len(angle)*dt, len(angle))`.
    - By default, line segment `capstyle` is set to "butt" to align segments smoothly.
    """

    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    x = np.asarray(angle)
    y = np.asarray(y)
    c = np.linspace(0, x.size*dt, x.size)
    
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    
    # correct the angel(x coordinate) if it jumps between 180 and -180 degree
    check1 = check2 = False
    for num in np.arange(len(segments)-1):
        if check1:
            segments[num][0][0] = segments[num-1][1][0] + 360
            check1 = False

        elif check2:
            segments[num][0][0] = segments[num-1][1][0] - 360
            check2 = False

        if segments[num][1][0] - segments[num+1][1][0] < -250:
            segments[num][2][0] = segments[num+1][1][0] -360
            check1 = True

        elif segments[num][1][0] - segments[num+1][1][0] > 250:
            segments[num][2][0] = segments[num+1][1][0]  + 360
            check2 = True

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def change_defaults(function, dictionary):
    
    """
    Dynamically change the default values of a function's parameters.

    Parameters
    ----------
    function : callable
        The function whose default parameter values will be modified.
    dictionary : dict
        Dictionary mapping parameter names to new default values.

    Notes
    -----
    - Only parameters that already have default values can be updated.
    - Parameters not specified in `dictionary` retain their original defaults.
    - This modifies the function in-place by updating `function.__defaults__`.
    - The order of defaults is preserved according to the original function signature.
    """

    original_defaults = [[name, param.default] for name, param in inspect.signature(function).parameters.items() 
                         if not type(param.default) == type]
    new_defaults = []
    for variable in original_defaults:
        if variable[0] in list(zip(*dictionary.items()))[0]:
            new_defaults.append(dictionary[variable[0]])
        else:
            new_defaults.append(variable[1])
    
    function.__defaults__ = tuple(new_defaults)
    
def dRMSD_from_descriptors(d, Connections_ref):
    
    """
    Compute the distance root-mean-square deviation (dRMSD) from a set of
    precomputed pairwise distances.

    Parameters
    ----------
    d : ndarray, shape (n_samples, n_distances)
        Array of distances for each sample.
    Connections_ref : ndarray, shape (n_distances, 3)
        Reference distances and atom indices. Column 2 contains the reference
        distance values (d0).

    Returns
    -------
    dRMSD : ndarray, shape (n_samples,)
        The computed dRMSD for each sample.

    Notes
    -----
    - The dRMSD is computed as:
        sqrt( sum((d - d0)^2) / (n-1) )
      where n is the number of distance connections.
    - This function assumes that `d` and `Connections_ref` are aligned such
      that each column in `d` corresponds to the respective reference distance.
    """

    d0 = Connections_ref[:, 2]
    n = len(Connections_ref)
    return np.sqrt(np.sum((d - d0) ** 2, axis=1) / (n-1))

def cartesian_to_spherical(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    
    # Avoid division by zero
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))

    return r, np.degrees(phi), np.degrees(theta)