from rdkit.Chem import Descriptors
from sklearn.preprocessing import scale
from mendeleev.fetch import fetch_ionization_energies, fetch_table
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from rdkit import Chem
from tqdm import tqdm
from rdkit import DataStructs

def get_fingerprint(df:pd.DataFrame):
    fp_numpy_list = []
    for idx, row in tqdm(df.iterrows(), desc = "Extract Fingerprint", total = len(df)):
        mol = Chem.MolFromSmiles(row["SMILES"])
        fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_numpy_list.append(arr)
    df['fp'] = fp_numpy_list
    return df

def load_dataset(
    train_dir = "./datasets/train.csv", 
    test_dir = "./datasets/test.csv", 
    submission_dir = "./datasets/sample_submission.csv",
    dev_dir = "./datasets/dev.csv"
    ):
    train_csv = pd.read_csv(train_dir)
    test_csv = pd.read_csv(test_dir)
    dev_csv = pd.read_csv(dev_dir)
    submission_csv = pd.read_csv(submission_dir)

    train_csv = pd.concat([train_csv, dev_csv])
    train_csv = train_csv.reset_index(drop = True)
    train_csv['ST1_GAP(eV)'] = train_csv['S1_energy(eV)'].values - train_csv['T1_energy(eV)'].values

    train_csv = get_fingerprint(train_csv)
    test_csv = get_fingerprint(test_csv)
    
    return train_csv, test_csv, submission_csv


# preprocessing

# Global Parameter
LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']
atom_feats = [7, 5, 4, 4, 2, 2, 4, 3, 8]
mol_feats = 22

def char2idx(x, set_num):
    if 0 <= x <= set_num: 
        return x
    else: 
        return 0

def atom2num(atom):
    '''Only use 7 atoms; ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']'''
    if atom == 'H':
        return 0
    elif atom in ['B', 'C', 'Si', 'Ge']:
        return 1
    elif atom in ['N', 'P']:
        return 2
    elif atom == 'O':
        return 3
    elif atom == 'F':
        return 4
    elif atom == 'S':
        return 5
    elif atom in ['Cl', 'Br', 'I']:
        return 6
    else:
        return -1


def hybrid2num(hybrid):
    if hybrid == 'SP3':
        return 1
    elif hybrid == 'SP2':
        return 2
    elif hybrid == 'SP':
        return 3
    else:
        return 0

def get_atom_table():
    atoms = np.array([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 35, 53])
    features = ['atomic_weight', 'atomic_radius', 'atomic_volume', 'electron_affinity',
                'dipole_polarizability', 'vdw_radius', 'en_pauling']
    elem_df = fetch_table('elements')
    feature_df = elem_df.loc[atoms-1, features]
    feature_df.set_index(atoms, inplace=True)
    ies = fetch_ionization_energies()
    final_df = pd.concat([feature_df, ies.loc[atoms]], axis=1)
    scaled_df = final_df.copy()
    scaled_df.iloc[:] = scale(final_df.iloc[:])
    return scaled_df

def get_mol_table(df):
    mol_properties = ['MolMR', 'NHOHCount', 'NOCount',
                      'NumHAcceptors', 'NumHDonors',
                      'NumHeteroatoms', 'NumValenceElectrons',
                      'MaxPartialCharge', 'MinPartialCharge',
                      'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
                      'NumAromaticHeterocycles', 'NumAromaticCarbocycles',
                      'NumSaturatedHeterocycles', 'NumSaturatedCarbocycles',
                      'NumAliphaticHeterocycles', 'NumAliphaticCarbocycles',
                      'RingCount', 'FractionCSP3', 'TPSA', 'LabuteASA']

    for idx, mol in tqdm(enumerate(df.loc[:, 'SMILES']), desc='Molecular Feature', total=len(df)):
        mol = Chem.MolFromSmiles(mol)
        for properties in mol_properties:
            df.loc[idx, properties] = getattr(Descriptors, properties)(mol)
    df.replace(np.inf, 0, inplace=True)
    df = df.fillna(0)
    scaled_df = df.copy()
    scaled_df.loc[:, mol_properties] = scale(df.loc[:, mol_properties])
    return scaled_df

def get_features(atom, table):
    features = []

    # Embedding Features
    features.append(char2idx(atom2num(atom.GetSymbol()), 6))
    features.append(char2idx(atom.GetDegree(), 4))
    features.append(char2idx(atom.GetTotalNumHs(), 3))
    features.append(char2idx(atom.GetImplicitValence(), 3))
    features.append(char2idx(int(atom.GetIsAromatic()), 1))
    features.append(char2idx(int(atom.IsInRing()), 1))
    features.append(char2idx(hybrid2num(atom.GetHybridization().name), 3))
    features.append(char2idx(atom.GetFormalCharge()+1, 2))

    # Continuous Features
    features += list(table.loc[atom.GetAtomicNum()].values)

    return features

from torch_geometric.data import Data

transform = T.Compose([
            T.ToUndirected(),
            T.Cartesian()
])

BOND_TYPE = {
    'ZERO': 0,
    'SINGLE': 1,
    'DOUBLE': 2,
    'TRIPLE': 3,
    'AROMATIC': 4,
    'IONIC': 5
}

def preprocessing(df : pd.DataFrame, mode = "train"):
    dataset = []
    atom_table = get_atom_table()

    for idx, row in tqdm(df.iterrows(), desc = "Atomic Feature", total = len(df)):
        mol = Chem.MolFromSmiles(row['SMILES'])
        adj = Chem.GetAdjacencyMatrix(mol)
        dist = Chem.GetDistanceMatrix(mol)

        # step 1 : get atom features from mol structure to index
        atom_features = []

        for atom in mol.GetAtoms():
            atom_features.append(get_features(atom, atom_table))
        
        atom_features = torch.tensor(atom_features, dtype = torch.float)

        # step 2 : get bond features from adjacency matrix
        bonds = []
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                if adj[i,j] == 1:
                    bonds.append([i,j])

        if len(bonds) == 0:
            continue

        bonds = torch.tensor(bonds, dtype = torch.long).t().contiguous()

        # step 3 : get edge attr from bond type

        bonds_attr = []

        for atom in mol.GetAtoms():
            # atom_idx = atom.GetIdx()
            for bond in atom.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                d = dist[start, end]
                bond_type = bond.GetBondType().name

                if bond_type not in BOND_TYPE.keys():
                    bond2idx = 0
                else:
                    bond2idx = BOND_TYPE[bond_type]

                bonds_attr.append([bond2idx, d])

        bonds_attr = torch.tensor(bonds_attr, dtype = torch.long).contiguous()
        
        if mode == "train":
            pred = row['ST1_GAP(eV)']
            pred = torch.tensor([pred], dtype = torch.float)
            dataset.append(Data(x = atom_features, edge_index=bonds, edge_attr = bonds_attr, y = pred, idx = idx))
        else:
            dataset.append(
                Data(x=atom_features, edge_index=bonds, edge_attr = bonds_attr, y = None, idx=idx))

    return dataset

from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

def generate_dataloader(
    train_dir="./datasets/train.csv",
    test_dir="./datasets/test.csv",
    submission_dir="./datasets/sample_submission.csv",
    dev_dir="./datasets/dev.csv",
    batch_size = 64,
    valid_ratio = 0.2
    ):

    train_csv, test_csv, submission_csv = load_dataset(train_dir, test_dir, submission_dir, dev_dir)

    datasets = preprocessing(train_csv, mode = 'train')
    test_datasets = preprocessing(test_csv, mode = 'test')

    total_indices = range(0,len(datasets))
    train_indices, valid_indices = train_test_split(total_indices, test_size = valid_ratio, random_state = 42, shuffle = True)
    
    train_loader = DataLoader(datasets, batch_size, sampler = SubsetRandomSampler(train_indices))
    valid_loader = DataLoader(datasets, batch_size, sampler = SubsetRandomSampler(valid_indices))
    test_loader = DataLoader(test_datasets, batch_size)

    return train_loader, valid_loader, test_loader

import matplotlib.pyplot as plt

def plot_training_curve(train_losses, valid_losses, save_dir = "./results/train-curve.png"):
    x_axis = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(x_axis, train_losses, 'ro--', label = "train loss")
    plt.plot(x_axis, valid_losses, 'b^--', label = "valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(save_dir)

