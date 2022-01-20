from rdkit import Chem
from tqdm import tqdm, trange

ONLY_LARGEST = True

def remove_atom_indices_from_mol(mol):
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)

def remove_atom_indices_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    remove_atom_indices_from_mol(mol)
    smi_cleaned = Chem.MolToSmiles(mol)
    return smi_cleaned


for dataset in ['train', 'test', 'val']:
    path = f'data/MIT_USPTO/{dataset}.txt'
    new_path = f'data/MIT_USPTO/{dataset}_cleaned.txt'
    write_lines = []

    tqdm_num_lines = sum(1 for line in open(path, 'r'))

    with open(path) as file:
        for i, line in enumerate(tqdm(file, total = tqdm_num_lines)):

            lhs, rhs = line.split()[0].split('>>')
            lhs_mols = lhs.split('.')
            rhs_mols = rhs.split('.')

            cleaned_lhs_mols = [] # mapping removed
            cleaned_rhs_mols = [] # mapping removed

            for lhs_mol_smi in lhs_mols:
                cleaned_lhs_mols.append(
                    remove_atom_indices_from_smi(lhs_mol_smi)
                )

            for rhs_mol_smi in rhs_mols:
                cleaned_rhs_mols.append(
                    remove_atom_indices_from_smi(rhs_mol_smi)
                )

            if ONLY_LARGEST:
                cleaned_lhs_mols = [max(cleaned_lhs_mols, key = len)]
                cleaned_rhs_mols = [max(cleaned_rhs_mols, key = len)]

            write_lines.append(
                '.'.join(cleaned_lhs_mols) + '>>' + '.'.join(cleaned_rhs_mols)
            )

    with open(new_path, 'w+') as f:

        for items in write_lines:
            f.write('%s\n' %items)

