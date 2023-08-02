import torch
from torch.utils.data import TensorDataset
from scipy.stats import maxwell
from Langevin import Langevin_Dyn
import os

class Data_Handler:
    """
    Class for handling data generation and loading.

    Parameters:
        Langevyn_args (dict): Arguments for initializing the Langevin_Dyn object (default is None).
        folder_path (str): Path to the folder where data will be saved (default is './data/').
        train_fraction (float): Fraction of data to be used for training (default is 0.9).
        validation_fraction (float): Fraction of training data to be used for validation (default is 0.1).
        device (str): Device to be used for loading the data (default is 'cpu').
    """

    def __init__(self, Langevyn_args=None, folder_path='./data/', train_fraction=0.9, validation_fraction=0.1, device='cpu'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for key in ['train', 'val', 'test']:
            if not os.path.exists(folder_path + 'data_{}/'.format(key)):
                os.makedirs(folder_path + 'data_{}/'.format(key))
        
        self.folder_path = folder_path
        if Langevyn_args is not None:
            self.dynamics = Langevin_Dyn(**Langevyn_args)
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.device = device

    def create_data_in_one_go(self, save_to_file=False):
        """
        Generates data as specified in the Langevin object and does a random splitting.
        Returns a dict of tensordatasets with keys '{train, val, test}_dataset', and a dict of separated tensors.
        """
        trajs, timeax, increments, A, B = self.dynamics.Data_generation()
        trajs = trajs.detach().to(torch.float32)
        timeax = timeax.to(torch.float32)
        increments = increments.to(torch.float32)
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        index = torch.randperm(trajs.size(0))
        train_size = int(len(index) * self.train_fraction)
        val_size = int(train_size * self.validation_fraction)
        train_index = index[val_size:train_size]
        val_index = index[0:val_size]
        test_index = index[train_size:]

        total_dict = {}
        iter_dict = {
            'trajs': trajs,
            'timeax': timeax,
            'increments': increments,
            'A': A,
            'B': B
        }

        for key in iter_dict.keys():
            train_tensor = iter_dict[key][train_index]
            val_tensor = iter_dict[key][val_index]
            test_tensor = iter_dict[key][test_index]

            total_dict['{}_train'.format(key)] = train_tensor
            total_dict['{}_val'.format(key)] = val_tensor
            total_dict['{}_test'.format(key)] = test_tensor

            if save_to_file:
                torch.save(train_tensor, self.folder_path + 'data_train/{}.pt'.format(key))
                torch.save(val_tensor, self.folder_path + 'data_val/{}.pt'.format(key))
                torch.save(test_tensor, self.folder_path + 'data_test/{}.pt'.format(key))

        tensordatasets_dict = {}

        for label in ['train', 'val', 'test']:
            tensors_list = [total_dict['{}_{}'.format(key, label)] for key in iter_dict.keys()]
            tensordata = TensorDataset(*tensors_list)
            tensordatasets_dict['{}_dataset'.format(label)] = tensordata

        return tensordatasets_dict, total_dict

    def load_datas_from_files(self):
        """
        Loads dataset from file structure.
        Returns a dict of tensordatasets with keys '{train, val, test}_dataset'.
        """
        tensordatasets_dict = {}

        for label in ['train', 'val', 'test']:
            tensors_list = [
                torch.load(self.folder_path + 'data_{}/{}.pt'.format(label, key), map_location=self.device) for key in ['trajs', 'timeax', 'increments', 'A', 'B']
            ]
            tensordata = TensorDataset(*tensors_list)
            tensordatasets_dict['{}_dataset'.format(label)] = tensordata

        return tensordatasets_dict
