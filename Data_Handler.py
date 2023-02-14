import torch
from torch.utils.data import TensorDataset
from scipy.stats import maxwell
from Langevin import Langevin_Dyn

class Data_Handler():
    
    def __init__(self, 
    Langevyn_args = None, 
    folder_path = './data/',
    device = 'cpu'
    ) -> None:

        self.folder_path = folder_path
        self.device = device
        
        if Langevyn_args is not None:

            if device != Langevyn_args['device']:
                raise TypeError("Specify the same device as the one used in Langevin_args. Found {} as opposed to {} in Langevin args".format(device, Langevyn_args['device']))

            self.dynamics = Langevin_Dyn(**Langevyn_args)
            
        
        

    def create_data_in_one_go(self, save_to_file = False):
        """
        Generates data as specified in the Langevin object and does a random splitting
        Returns a dict of tensordatasets with keys '{train, val, test}_dataset', and a dict of separated tensors
        """
        trajs, timeax, increments, B = self.dynamics.Data_generation()
        trajs = trajs.detach()

        index = torch.randperm(trajs.size(0))
        train_size = int(len(index)*0.7)
        val_size = int(train_size*0.1)
        train_index = index[val_size:train_size]
        val_index = index[0:val_size]
        test_index = index[train_size::]

        total_dict = {}

        iter_dict = {
            'trajs': trajs,
            'timeax': timeax,
            'increments': increments,
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
            tensordatasets_dict['{}_dataset'] = tensordata

        return tensordatasets_dict, total_dict

    

    def load_datas_from_files(self):
        
        """
        Loads dataset from file structure
        Returns a dict of tensordatasets with keys '{train, val, test}_dataset'
        """

        tensordatasets_dict = {}

        for label in ['train', 'val', 'test']:

            tensors_list = [torch.load(self.folder_path + 'data_{}/{}.pt'.format(label, key))for key in ['trajs', 'timeax', 'increments', 'B']]
            tensordata = TensorDataset(*tensors_list)
            tensordatasets_dict['{}_dataset'.format(label)] = tensordata
        
        return tensordatasets_dict
