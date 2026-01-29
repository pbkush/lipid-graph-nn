from pathlib import Path
import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import training_nn_multiple as custom
from tqdm import tqdm
from functions import pkl_load, pkl_save, mkdir
import itertools
from sklearn.preprocessing import StandardScaler
import properties_nn as properties_nn


class feed_forward(torch.nn.Module):
    
    """
    Feed-forward neural network for modeling the free energy landscape.

    This class constructs a fully connected neural network with optional
    batch normalization, dropout, and custom activation functions for each
    hidden layer. The network outputs a scalar free energy for each input
    descriptor vector.

    Args:
        n_features (int, default=3136):
            Number of input features (size of descriptor vector).
        hidden_layers (list of int, default=[512, 512, 512, 512]):
            Number of neurons in each hidden layer.
        activation (torch.nn.Module or list of modules, default=torch.nn.PReLU(512)):
            Activation function(s) to apply after each hidden layer.
        dropout (float, default=0.0):
            Dropout probability applied after each hidden layer.
        batch_norm (bool, default=False):
            Whether to apply batch normalization after each hidden layer.
        scaler (optional):
            Optional pre-processing scaler to normalize input descriptors.

    Attributes:
        net (torch.nn.Sequential):
            Sequential model containing the fully connected layers with
            activations, batch normalization, and dropout applied.
        input_parameters (dict):
            Dictionary storing the initialization parameters for reproducibility.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the free energy given input descriptors.
    """

    
    def __init__(self,
                 n_features:int= 3136,
                 hidden_layers:list[int]=[512, 512, 512, 512],
                 activation=torch.nn.PReLU(512),
                 dropout:float=0.0,
                 batch_norm:bool=False,
                 scaler=None):

        self.scaler = scaler
        self.input_parameters = {'n_features'   : n_features,
                                 'hidden_layers': hidden_layers,
                                 'activation'   : activation,
                                 'dropout'      : dropout,
                                 'batch_norm'   : batch_norm,
                                 'scaler'       : scaler}
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
        return torch.squeeze(self.net(x))
    

def train_RNN(network,
          epochs,
          optimizer,
          train_dataloader,
          test_dataloader,
          device,
          dtype,
          stop=30,
          save_to='best_model.h5',
          verbose=True):
    
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
    
    train_losses, test_losses, scales = [], [], []
    min_loss = mean_error = max_error = 1e10
    count = 0
    count_flag = False

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    lr_scale = np.linspace(10, 0.1, epochs)
    lr = optimizer.param_groups[0]['lr']
    for epoch in pbar:
        
        for param_group in optimizer.param_groups:
            # slowly decrease lr
            param_group['lr'] = lr * lr_scale[epoch]
            
        # TRAIN
        network.train()
        
        batch_losses = []
        for batch_idx, (descriptors, results) in enumerate(train_dataloader):
            
            # Move data to the device
            descriptors = descriptors.to(device=device, dtype=dtype)
            results = results.to(device=device, dtype=dtype)
            network_estimate = network(descriptors)
            # Define the closure function
            def closure():
                optimizer.zero_grad()
                
                # Define loss function
                loss_fn = torch.nn.MSELoss()

                # Compute loss
                depth_free_energy = (torch.abs(torch.squeeze(results)))+1
                error = torch.abs(torch.squeeze(network_estimate) - torch.squeeze(results))
                loss = ((error**2)*depth_free_energy).sum()
                loss.backward()
                return loss
            
            # Perform optimization step
            loss = optimizer.step(closure)

            # Record metrics
            batch_losses.append(float(loss)/len(descriptors))
        
        train_losses.append(np.mean(batch_losses))
        
        # TEST
        network.eval()
        
        batch_test_losses = []
        error_list = []
        for batch_idx, (descriptors, results) in enumerate(test_dataloader):
            descriptors = descriptors.to(device=device, dtype=dtype)
            results = results.to(device=device, dtype=dtype)
            network_estimate = network(descriptors)
            loss_fn = torch.nn.MSELoss()

            # Compute loss
            bin_center = ((descriptors[:, 1] + descriptors[:, 2])/2)**-1
            depth_free_energy = (torch.abs(torch.squeeze(results)))+1
            error = torch.abs(torch.squeeze(network_estimate) - torch.squeeze(results))
            loss = ((error**2)*depth_free_energy).sum()
            
            # get average of the single batch (32 data points
            batch_test_losses.append(float(loss)/len(descriptors))
            error_list.append(error.cpu().detach().numpy())
            
        # get average of the batches
        test_losses.append(np.mean(batch_test_losses))
        error_list = np.concatenate(error_list)
        
        # save the model if the test loss is the lowest
        if test_losses[-1] < min_loss and train_losses[-1] < 25:
            min_loss = test_losses[-1]
            pkl_save(save_to, [network.input_parameters, network.state_dict()])
            count_flag = True
            if test_losses[-1] < 20:
                break

        if epoch % 10 == 0:
            pbar.set_postfix({'Epoch':epoch,
                      'Train loss':"%.3g" %train_losses[-1],
                      'validation loss':np.round(test_losses[-1], 6),
                      'best validation loss': min_loss,
                             'mean_error':np.mean(error_list),
                             'max_error':np.max(error_list)}) #np.round(np.min(test_losses), 6)})
    if not count_flag:
        pkl_save(save_to, [network.input_parameters, network.state_dict()])

    return train_losses, test_losses

def evaluate(network, descriptors, batch_size=4096):
    
    """
    Evaluates a neural network model using PyTorch.

    Args:
        network (torch.nn.Module): The neural network to evaluate.
        descriptors (numpy.ndarray): The input data for evaluation.
        device (torch.device): The device on which to perform the evaluation.
        dtype (torch.dtype): The data type for input tensors.
        batch_size (int, optional): The batch size for evaluation.
                                    Default is 4096.

    Returns:
        numpy.ndarray: The network output as a NumPy array.
    """
    
    # initialize
    results = []
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    network.eval()
    
    # compute in batches
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            descriptors, batch_size=batch_size, shuffle=False):
            batch = batch.to(device=device, dtype=dtype)
            output = network(batch).detach().cpu().numpy().ravel()
            results.append(output)
    
    # return
    if len(results):
        return np.concatenate(results)
    else:
        return np.zeros(0)
    
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


class feed_forward_dataset(Dataset):
    
    """
    PyTorch Dataset for preparing feed-forward neural network training
    from simulation data.

    This dataset class wraps descriptors and corresponding results
    for training feed-forward neural networks. It supports
    filtering by membrane compositions and bin shifts, and
    automatically handles missing or invalid data.

    Args:
        data (dict):
            Nested dictionary containing descriptors and reference results
            organized by membrane composition and bin shift.
        membrane_compositions (list, optional):
            List of membrane compositions to include. If None, uses all keys in `data`.
        bin_numbers (list, optional):
            List of bin numbers (not used directly in this implementation).
        bin_shift (list, optional):
            List of shifts for binning the data. Defaults to all shifts for the first composition.
        iterator (list, optional):
            Specifies the order of compositions to iterate over. Defaults to `membrane_compositions`.

    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Returns the descriptors and results for the sample at the given index as torch tensors.
        generate(membrane_compositions=None, bin_shift=None, iterator=None):
            Internal method to populate the dataset from `data`, filtering by valid entries.
    """

    def __init__(self,
                 data,
                 membrane_compositions:list=None,
                 bin_numbers:list=None,
                 bin_shift:list=None, 
                 iterator=None):
        """
        Args:
            npz_file (str): Path to the .npz file containing descriptors, shooting_results, and weights.
        """
        # cut the 
        self.data = data
        self.generate(membrane_compositions=membrane_compositions,
                      bin_shift=bin_shift,
                      iterator=iterator)
    
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
        descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float32)
        results = torch.tensor(self.results[idx], dtype=torch.float32)
        return descriptors, results
    
    def generate(self, membrane_compositions=None, bin_shift:list=None, iterator=None):
        
        if not membrane_compositions:
            membrane_compositions = list(self.data.keys())
            
        if not bin_shift:
            bin_shift = list(self.data[membrane_compositions[0]].keys())
        
        if not iterator:
            iterator = membrane_compositions
        
            
        descriptors = []
        results = []
        
        for composition in iterator:
            for shift in bin_shift:
                input_composition = self.data[composition][shift]
                results_composition = input_composition['reference'][0]
                descriptors_composition = input_composition['descriptors']
                mask = np.isfinite(results_composition)
                if len(results_composition[mask]) == 0:
                    continue

                descriptors.append(descriptors_composition[mask, :])
                results.append(results_composition[mask])

        self.descriptors = np.concatenate(descriptors)
        self.results = np.reshape(np.concatenate(results), (-1,1))
        
class RNNArchLSTM(torch.nn.Module):
    
    """
    Recurrent Neural Network architecture using LSTM layers.

    This class defines an LSTM-based recurrent neural network followed by
    a dense (fully connected) layer to output predictions. It is designed
    for sequence-based input data and can be used to model temporal
    dependencies in descriptor trajectories.

    Args:
        size_input (int):
            Number of features in each input timestep.
        size_hidden (int):
            Number of hidden units in each LSTM layer.
        numb_RNN_layers (int):
            Number of stacked LSTM layers.
        size_output (int):
            Number of output features (e.g., scalar free energy or property).
        dropout (float, default=0):
            Dropout probability applied between LSTM layers.
        train_on (list, default=['']):
            List of labels or identifiers indicating which properties the model trains on.
        device (torch.device, default=torch.device('cpu')):
            Device to place the model and tensors on.

    Attributes:
        recur_layer_set (torch.nn.LSTM):
            Stacked LSTM layers for sequence processing.
        dense_layer_set (torch.nn.Linear):
            Fully connected layer to map LSTM outputs to final predictions.
        input_parameters (dict):
            Dictionary storing initialization parameters for reproducibility.

    Methods:
        forward(X: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the LSTM and dense layers.
            Args:
                X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, size_input).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, size_output)
                or squeezed to remove single-dimensional entries.
    """

    def __init__(self, 
                 size_input,
                 size_hidden,
                 numb_RNN_layers,
                 size_output,
                 dropout= 0,
                 train_on=[''],
                 device = torch.device('cpu')):
        
        super(RNNArchLSTM, self).__init__()
        
        self.D = size_input
        self.M = size_hidden
        self.L = numb_RNN_layers
        self.K = size_output
        self.device = device
        self.train_on = train_on
        self.input_parameters = {'size_input'   : size_input,
                                 'size_hidden': size_hidden,
                                 'numb_RNN_layers'   : numb_RNN_layers,
                                 'size_output'      : size_output,
                                 'train_on':train_on}
        
        
        # RECUR LAYER SET [START]
        self.recur_layer_set = torch.nn.LSTM(
            input_size = self.D,
            hidden_size = self.M,
            num_layers = self.L,
            dropout= dropout,
#             proj_size = 1,
            batch_first = True
        )
        
        # DENSE LAYER SET [FINAL]
        self.dense_layer_set = torch.nn.Linear(in_features = self.M, out_features = self.K)
        
    def forward(self, X):

        h_0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)
        c_0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)
        Y, _ = self.recur_layer_set(X, (h_0, c_0))
        Y = self.dense_layer_set(Y[:, :, :])
        return torch.squeeze(Y)
    
    
class RNN_dataset(Dataset):
    
    """
    PyTorch Dataset for preparing RNN training from simulation data.

    This dataset class wraps descriptors and corresponding results
    for training recurrent neural networks (RNNs). It supports
    filtering by membrane compositions and bin shifts, and
    automatically handles missing or invalid data.

    Args:
        data (dict):
            Nested dictionary containing descriptors and reference results
            organized by membrane composition and bin shift.
        membrane_compositions (list, optional):
            List of membrane compositions to include. If None, uses all keys in `data`.
        bin_numbers (list, optional):
            List of bin numbers (not used directly in this implementation).
        bin_shift (list, optional):
            List of shifts for binning the data. Defaults to all shifts for the first composition.
        iterator (list, optional):
            Specifies the order of compositions to iterate over. Defaults to `membrane_compositions`.

    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Returns the descriptors and results for the sample at the given index as torch tensors.
        generate(membrane_compositions=None, bin_shift=None, iterator=None):
            Internal method to populate the dataset from `data`, filtering by valid entries.
    """

    def __init__(self,
                 data,
                 membrane_compositions:list=None,
                 bin_numbers:list=None,
                 bin_shift:list=None, 
                 iterator=None):
        """
        Args:
            npz_file (str): Path to the .npz file containing descriptors, shooting_results, and weights.
        """
        # cut the 
        self.data = data
        self.generate(membrane_compositions=membrane_compositions,
                      bin_shift=bin_shift,
                      iterator=iterator)
    
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
        descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float32)
        results = torch.tensor(self.results[idx], dtype=torch.float32)
        return descriptors, results
    
    def generate(self, membrane_compositions=None, bin_shift:list=None, iterator=None):
        
        if not membrane_compositions:
            membrane_compositions = list(self.data.keys())
            
        if not bin_shift:
            bin_shift = list(self.data[membrane_compositions[0]].keys())
        
        if not iterator:
            iterator = membrane_compositions
        
            
        descriptors = []
        results = []
        
        for composition in iterator:
            for shift in bin_shift:
                input_composition = self.data[composition][shift]
                results_composition = input_composition['reference'][0]
                descriptors_composition = input_composition['descriptors']
                mask = np.isfinite(results_composition)
                if len(results_composition[mask]) == 0:
                    continue

                descriptors.append(descriptors_composition[mask, :])
                results.append(results_composition[mask])        
        self.descriptors = descriptors
        self.results = results
        
def values_function_combined_nn(descriptors,
                                network,
                                property_model,
                                composition,
                                scaler=None,
                                connection_mask=None):
    
    """
    Compute property values using a neural network and a property model for use in 
    AIMMD computation.

    This function evaluates a neural network on descriptors of a system,
    optionally scaled, and augmented with additional properties derived from
    a separate property model and the system's composition.

    Args:
        descriptors (np.ndarray):
            Array of system descriptors to evaluate (shape: n_samples x n_features).
        network (torch.nn.Module):
            Trained neural network for predicting the target values.
        property_model:
            Model used to compute additional properties from the system composition.
        composition (list or np.ndarray):
            Composition information of the system to compute additional properties.
        scaler (sklearn.preprocessing, optional):
            Scaler to transform descriptors before passing to the network.
        connection_mask (np.ndarray or list, optional):
            Boolean or integer mask to select a subset of descriptor features.

    Returns:
        np.ndarray:
            Array of predicted values from the network for each input descriptor.
    """
    
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    network.eval()
    
    # initialize
    results = []
    reduced_descriptors = descriptors[:, connection_mask]
    properties = properties_nn.evaluate(property_model, [properties_nn.transfer_membrane_comp(composition)])
    properties = np.repeat(np.array([properties]), reduced_descriptors.shape[0], axis=0)
    new_descriptors = np.hstack((reduced_descriptors, properties))
    if scaler:
        new_descriptors = scaler.transform(new_descriptors)
    # compute in batches
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            new_descriptors, batch_size=4096, shuffle=False):
            batch = batch.to(device=device, dtype=dtype)
            output = network(batch).detach().cpu().numpy().ravel()
            results.append(output)
    
    # return
    return np.concatenate(results)
