import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem

dataFolder = 'data/external'

def swish(x):
    return x * torch.sigmoid(x)

class simpleNet(nn.Module):

    def __init__(self, representation):

        super(simpleNet, self).__init__()

        if representation == 'fgroup':
            fgroups = pd.read_csv(dataFolder + '/Functional_groups_filtered.csv')
            fgroups_list = list(map(lambda x: Chem.MolFromSmarts(x), fgroups['SMARTS'].tolist()))

            #! install new rdkit to skip this step
            fgroups_list = [x for x in fgroups_list if x is not None]

            rep_length = len(fgroups_list)

        elif representation == 'chembl':
            vocab_map = pd.read_csv(dataFolder + '/subword_units_map_drug_chembl_1500.csv')
            rep_length = len(vocab_map)

        self.fc1 = nn.Linear(rep_length, 5000)
        self.bn1 = nn.BatchNorm1d(5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.bn2 = nn.BatchNorm1d(5000)
        self.fc3 = nn.Linear(5000, rep_length)

    def forward(self, x):

        x = swish(self.fc1(x))
        x = self.bn1(x)
        x = swish(self.fc2(x))
        x = self.bn2(x)
        x = torch.sigmoid(self.fc3(x))

        return x
