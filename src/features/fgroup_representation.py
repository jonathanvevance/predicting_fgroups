import numpy as np
import pandas as pd
from rdkit import Chem

dataFolder = 'data/external'
fgroups = pd.read_csv(dataFolder + '/Functional_groups_filtered.csv')                   # extracted from ochem (web crawling) for toxicity detection
fgroups_list = list(map(lambda x: Chem.MolFromSmarts(x), fgroups['SMARTS'].tolist()))   # SMARTS = patterns derived from SMILES

#! install new rdkit to skip this step
fgroups_list = [x for x in fgroups_list if x is not None] #! length = 2742 (original = 2786)

def get_rep_from_mol_list(mol_list):

    rep_vector = np.zeros(len(fgroups_list),)

    for mol_smi in mol_list:

        molecule = Chem.MolFromSmiles(mol_smi)

        for idx in range(len(fgroups_list)):

            if rep_vector[idx] == 1:
                continue

            if molecule.HasSubstructMatch(fgroups_list[idx]):
                rep_vector[idx] = 1

    return rep_vector # multi hot vector
