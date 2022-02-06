
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data

from features.chembl_representation import get_rep_from_mol_list as get_chembl_rep
from features.fgroup_representation import get_rep_from_mol_list as get_fgroup_rep


cleaned_data_paths_dict = {
    'train' : 'data/interim/train_cleaned.txt',
    'val' : 'data/interim/val_cleaned.txt',
    'test' : 'data/interim/test_cleaned.txt'
}

processed_data_paths_dict = {
    'chembl': {
        'train' : 'data/processed/chembl/train_processed.npy',
        'val' : 'data/processed/chembl/val_processed.npy',
        'test' : 'data/processed/chembl/test_processed.npy'
    },
    'fgroup': {
        'train' : 'data/processed/fgroup/train_processed.npy',
        'val' : 'data/processed/fgroup/val_processed.npy',
        'test' : 'data/processed/fgroup/test_processed.npy'
    }
}

class reactionDataset(data.Dataset):

    def __init__(
        self,
        mode = 'train',
        features_data = None,
        scaler = None,
        representation = 'fgroup'
    ):

        processed_dataset_path = processed_data_paths_dict[representation][mode]

        if os.path.exists(processed_dataset_path):

            # https://newbedev.com/saving-dictionary-of-numpy-arrays
            dataset_dict = np.load(processed_dataset_path, allow_pickle = True)
            self.X = dataset_dict[()]['X']
            self.Y = dataset_dict[()]['Y']

        else:

            #! REMOVE
            count = 0
            class_freq = None

            self.X = []
            self.Y = []
            self.features_data = features_data
            self.scaler = scaler

            if representation == 'chembl':
                get_rep_from_mol_list = get_chembl_rep
            elif representation == 'fgroup':
                get_rep_from_mol_list = get_fgroup_rep
            else:
                raise NotImplementedError

            cleaned_dataset_path = cleaned_data_paths_dict[mode]
            tqdm_num_lines = sum(1 for line in open(cleaned_dataset_path, 'r'))

            with open(cleaned_dataset_path) as file:
                for idx, line in enumerate(tqdm(file, total = tqdm_num_lines)):

                    # if idx == 100:
                    #     break # check if it is learning

                    lhs, rhs = line.split()[0].split('>>')
                    lhs_mols = lhs.split('.')
                    rhs_mols = rhs.split('.')

                    try:
                        x_vec = get_rep_from_mol_list(lhs_mols)
                        y_vec = get_rep_from_mol_list(rhs_mols)
                    except:
                        continue # chembl-bpe has some issues

                    if self.features_data is not None:
                        # x_vec = np.concatenate([x_vec, self.features_data[idx]])
                        raise NotImplementedError # have to standardize indices

                    if self.scaler is not None:
                        raise NotImplementedError # multi hot vectors dont need scaling

                    #! UNCOMMENT
                    # self.X.append(x_vec)
                    # self.Y.append(y_vec)

                    #! REMOVE
                    count += 1
                    if class_freq is None:
                        class_freq = y_vec
                    else:
                        class_freq += y_vec

            self.X = np.array(self.X)
            self.Y = np.array(self.Y)

            # save pickle for the future # https://newbedev.com/saving-dictionary-of-numpy-arrays
            dataset_dict = {'X': self.X, 'Y': self.Y}
            # np.save(processed_dataset_path, dataset_dict) # pickling issue with large files

            #! REMOVE
            class_freq /= count
            total = np.size(class_freq)
            sufficient = np.count_nonzero(np.where(class_freq > 0.05)) # imbalance that is not worse than 95%
            print(total, total - sufficient, sufficient)
            exit()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.Y[index])

    def get_class_weights(self):
        non_zero_count = np.count_nonzero(self.Y)
        zero_count = self.Y.size - non_zero_count
        non_zero_frac = non_zero_count / (non_zero_count + zero_count)
        zero_frac = 1 - non_zero_frac
        return torch.Tensor([non_zero_frac, zero_frac])
