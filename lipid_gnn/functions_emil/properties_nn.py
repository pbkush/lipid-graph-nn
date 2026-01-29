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
from tqdm import tqdm
from scipy.special import logit, expit
# from utils import evaluate
import matplotlib.pyplot as plt
import matplotlib.animation as animationt
from IPython import display
from time import sleep
from functions import pkl_save, pkl_load

class Network(torch.nn.Module):
    """Neural network representation of the free energy landscape."""
    
    def __init__(self,
                 n_features:int= 10,
                 output_size= 7, 
                 hidden_layers:list[int]=[512],
                 activation=[torch.nn.PReLU(512)],
                 dropout:float=0.0,
                 batch_norm:bool=False):
        """
        Initialize a neural network to represent the free energy landscape.
        
        Parameters:
        -----------
        hidden_layers : List[int]
            List of hidden layer sizes
        activation : torch.nn.Module
            Activation function to use between layers
        """
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
        layers.append(torch.nn.Linear(prev_size, output_size))
        
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
    def __init__(self, 
                 data, 
                 train_on = ['CHOL', 'DIPC', 'DOPC', 'DOPE', 'DOPS', 'DPPC', 'DPPE', 'POPC','POPE', 'POPS'],
                 iterator=None,
                 verbose = True):
        """
        Args:
            npz_file (.npz-file): .npz-file containing descriptors, results, and weights 
                                  from shootings, equilibriumA and equilibriumB conformations.
        """
        # Load the .npz file
        self.membrane_compositions     = data['membrane_compositions'],
        self.membrane_property_values  = data['membrane_property_values']
        self.membrane_properties       = data['membrane_properties']
        self.train_on                  = train_on
        
        self.get(iterator=iterator)
        
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
    
    def get(self, train_on=None, iterator=None):
        """
        Returns a single sample of data.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (features, label, weight) for the given index.
        """   
        # put everything together
        if not train_on:
            train_on = self.train_on
        else:
            self.train_on = train_on
        
        if not iterator:
            iterator = self.membrane_compositions[0]
        
        descriptors = []
        results = []
        for composition in iterator:
            index_array = np.where(self.membrane_compositions[0] == composition)[0]
            if index_array.size != 1:
                print(f'something is wrong with {composition}')
                continue
            index = index_array[0]
            descriptors.append(transfer_membrane_comp(composition, train_on))
            results.append(self.membrane_property_values[index][[0,1,2,3,5,6,7]])
        self.descriptors = np.asanyarray(descriptors)
        self.results = np.asanyarray(results)






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

def train(network,
          epochs,
          batch_size,
          optimizer,
          train_dataset,
          test_dataset,
          device,
          dtype,
          stop=30,
          save_to='best_model.h5',
          min_train_loss=np.inf,
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
    train_losses, test_losses= [], []
    min_loss = max_error_prop = max_error = condition = check =  1e10
    count = 0
    count_flag = False

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    for epoch in pbar:
        
        # TRAIN
        network.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        batch_losses = []
        for batch_idx, (descriptors, results) in enumerate(train_dataloader):
            
            # Move data to the device
            descriptors = descriptors.to(device=device, dtype=dtype)
            results = results.to(device=device, dtype=dtype)
            network_estimate = network(descriptors)
            # Define the closure function
            def closure():
                optimizer.zero_grad()
                
                # loss
                sign = torch.abs(torch.sign(network_estimate)-2)
                loss = (((torch.abs(network_estimate - results)*results**-1)*100)*sign).sum()
                loss.backward()
                return loss
            
            # Perform optimization step
            loss = optimizer.step(closure)
            
            # Record metrics
            batch_losses.append(float(loss)/len(network_estimate))
        
        train_losses.append(np.mean(batch_losses))
        
        # TEST
        network.eval()

            
        if epoch%20 == 0:
            error = []
            error_prop = []
            for membrane_composition in os.listdir('pickles/properties'):
                estimate = evaluate(network, np.array([transfer_membrane_comp(membrane_composition[:-3])]))
                properties, values = zip(*pkl_load(f'pickles/properties/{membrane_composition}')[0].items())
                result = np.asanyarray(values)[[0,1,2,3,5,6,7]]
                error.append(np.mean(np.abs(result - estimate)/result)*100)
                error_prop.append(np.max(np.abs(result - estimate)/result)*100)
            max_error = np.max(error)
            max_error_prop = np.max(error_prop)
            condition = max_error*2 + max_error_prop
            pbar.set_postfix({'Epoch':epoch,
                      'Train loss':"%.3g" %train_losses[-1],
#                       'validation loss':np.round(test_losses[-1], 6),
                      'best validation loss': check,
                             'max_error':max_error,
                             'max_error_prop':max_error_prop}) #np.round(np.min(test_losses), 6)})
            if condition < min_loss:
                min_loss = condition
                check = [np.round(max_error, 2), np.round(max_error_prop, 2)]
                pkl_save(save_to, [network.input_parameters, network.state_dict()])
        

    return train_losses, test_losses

def train_sweep(network,
          epochs,
          batch_size,
          optimizer,
          train_dataset,
          test_dataset,
          device,
          dtype,
          stop=30,
          save_to='best_model.h5',
          min_train_loss=np.inf,
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
    train_losses, test_losses= [], []
    min_loss = max_error_prop = max_error = min_max_error = min_max_error_prop = 1e10
    count = 0
    count_flag = False

    pbar = tqdm(range(epochs), disable=not verbose)
    pbar.bar_format = '{desc:}{postfix}'
    for epoch in pbar:
        
        # TRAIN
        network.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
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
                sign = torch.abs(torch.sign(network_estimate)-2)
                loss = (((torch.abs(network_estimate - results)*results**-1)*100)*sign).sum()
                loss.backward()
                return loss
            
            # Perform optimization step
            loss = optimizer.step(closure)
            
            # Record metrics
            batch_losses.append(float(loss))
        
        train_losses.append(np.mean(batch_losses))
        
        # TEST
        network.eval()
        
        # save the model if the test loss is the lowest
        if max_error_prop < min_loss:
            min_loss = max_error_prop
            min_max_error_prop = max_error_prop
            min_max_error = max_error
            pkl_save(save_to, [network.input_parameters, network.state_dict()])
            
        if epoch%20 == 0:
            error = []
            error_prop = []
            for membrane_composition in os.listdir('membrane_compositions'):
                estimate = evaluate(network, np.array([transfer_membrane_comp(membrane_composition)]))
                properties, values = zip(*pkl_load(f'properties/{membrane_composition}.h5')[0].items())
                result = np.asanyarray(values)
                error.append(np.mean(np.abs(result - estimate)/result)*100)
                error_prop.append(np.max(np.abs(result - estimate)/result)*100)
            max_error = np.max(error)
            max_error_prop = np.max(error_prop)
            pbar.set_postfix({'Epoch':epoch,
                      'Train loss':"%.3g" %train_losses[-1],
                      'best validation loss': [min_max_error, min_max_error_prop]}) #np.round(np.min(test_losses), 6)})
        

    return train_losses, test_losses, [min_max_error, min_max_error_prop]


    
def transfer_membrane_comp(membrane_composition:str,
                           train_on:list=['CHOL', 'DIPC', 'DOPC', 'DOPE', 'DOPS', 'DPPC', 'DPPE', 'POPC','POPE', 'POPS']):
    
    """
    Convert a membrane composition string into a numerical array.

    This function parses a string describing the membrane composition
    (e.g., "POPC20_DPPC30_CHOL50") and maps it to a normalized array
    representing the fraction of each lipid type specified in `train_on`.

    Args:
        membrane_composition (str):
            String specifying the membrane composition, where each lipid
            is followed by its percentage (e.g., "POPC20_DPPC30_CHOL50").
        train_on (list of str, optional):
            List of lipid types to include in the output array. Default
            includes common lipids.

    Returns:
        np.ndarray:
            Array of length `len(train_on)` with normalized lipid fractions
            (values divided by 10).
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

def create_dataset(raw_data_dir:(pathlib.PosixPath, str),
                   save:str     = 'dataset.npz',
                   verbose:bool = True):
    '''
    Create dataset from raw data like shooting points and shooting results
    
    raw_data_dir : path to directory with raw data 
    train_on     : lipids or membrane compositions you want to train on
    save         : path you want to save the dataset
    verbose      : print out stuff
    '''
    
    # convert path into PosixPath if necessary
    if isinstance(raw_data_dir, str):
        raw_data_dir = Path(raw_data_dir)
    
    results_dict = {'membrane_compositions':[],
                    'membrane_property_values':[]}
    # Main loop
    for path in raw_data_dir.glob('*'):
        
        membrane_composition = path.stem
        membrane_properties, values = zip(*pkl_load(path)[0].items())
        
        results_dict['membrane_compositions'].append(membrane_composition)
        results_dict['membrane_property_values'].append(values)
    results_dict['membrane_properties'] = membrane_properties
    results_dict['membrane_compositions'] = np.asanyarray(results_dict['membrane_compositions'])
    results_dict['membrane_property_values'] = np.asanyarray(results_dict['membrane_property_values'])     
        
    # save results in npz file
    np.savez(save, **results_dict)
    
    # print out size of dataset
    if verbose:
        print(f'membrane_compositions:    {results_dict["membrane_compositions"].shape} \n'
              f'membrane_property_values: {results_dict["membrane_property_values"].shape} \n'
              f'membrane_properties:      {results_dict["membrane_properties"]}')
    return np.load(save)

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
