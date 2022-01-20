import numpy as np
import pandas as pd

from subword_nmt.apply_bpe import BPE
import codecs

dataFolder = 'data/external'
vocab_path = dataFolder + '/codes_drug_chembl_1500.txt'
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='')

vocab_map = pd.read_csv(dataFolder + '/subword_units_map_drug_chembl_1500.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word))))

def smiles2index(s1):
    t1 = bpe.process_line(s1).split() #split
    i1 = [words2idx[i] for i in t1] #index
    return i1

def index2multi_hot(i1):
    v1 = np.zeros(len(idx2word),)
    v1[i1] = 1
    return v1

def smiles2vector(s1):
    i1 = smiles2index(s1)
    v_d = index2multi_hot(i1)
    return v_d

def get_rep_from_mol_list(mol_list):

    rep_vector_list = []

    for mol_smi in mol_list:
        try:
            rep_vector = smiles2vector(mol_smi)
            rep_vector_list.append(rep_vector)
        except:
            pass # skip this molecule

    rep_vector = np.max(rep_vector_list, axis = 0) # max across columns

    return rep_vector # multi hot vector
