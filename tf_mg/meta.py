from rdkit import Chem
from os import path


ATOM_SYMBOLS = ['C', 'F', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S']
BOND_ORDERS = [Chem.BondType.AROMATIC,
               Chem.BondType.SINGLE,
               Chem.BondType.DOUBLE,
               Chem.BondType.TRIPLE]

def get_atom_type(atom):
    atom_symbol = ATOM_SYMBOLS.index(atom.GetSymbol())
    atom_charge = atom.GetFormalCharge()
    atom_hs = atom.GetNumExplicitHs()
    return [atom_symbol, atom_charge, atom_hs]

def get_bond_type(bond):
    return BOND_ORDERS.index(bond.GetBondType())

def _build_atom_types():
    atom_types = []
    with open('data/atom_types.txt', 'w') as f:
        with open('data/ChEMBL_cleaned.txt') as h:
            for line in h:
                if line == '':
                    break
                smiles = line.strip('\n').strip('\r')
                m = Chem.MolFromSmiles(smiles)
                for a in m.GetAtoms():
                    atom_type = get_atom_type(a)
                    if atom_type not in atom_types:
                        atom_types.append(atom_type)
        for atom_symbol, atom_charge, atom_hs in atom_types:
            f.write(str(atom_symbol) + ',' + str(atom_charge) + ',' + str(atom_hs) + '\n')

def _load_atom_types():
    atom_types = []
    with open('data/atom_types.txt') as f:
        for line in f:
            atom_types.append([int(x) for x in line.strip('\n').split(',')])
    return atom_types

if not path.isfile('data/atom_types.txt'):
    _build_atom_types()

ATOM_TYPES = _load_atom_types()
BOND_TYPES = range(len(BOND_ORDERS))

NUM_ATOM_TYPES = len(ATOM_TYPES)
NUM_BOND_TYPES = len(BOND_TYPES)

def atom_to_index(atom):
    return ATOM_TYPES.index(get_atom_type(atom))

def bond_to_index(bond):
    return get_bond_type(bond)

def index_to_atom(idx):
    atom_symbol, atom_charge, atom_hs = ATOM_TYPES[idx]
    a = Chem.Atom(ATOM_SYMBOLS[atom_symbol])
    a.SetFormalCharge(atom_charge)
    a.SetNumExplicitHs(atom_hs)
    return a

def index_to_bond(mol, begin_id, end_id, idx):
    mol.AddBond(begin_id, end_id, BOND_ORDERS[idx])

