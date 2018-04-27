from abc import ABCMeta, abstractmethod

import numpy as np


__all__ = ['Conditional', 'Delimited', 'SparseFP', 'ScaffoldFP']

class Conditional(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Delimited(Conditional):

    def __init__(self, d='\t'):
        self.d = d

    def __call__(self, line):
        line = line.strip('\n').strip('\r')
        line = line.split(self.d)

        smiles = line[0]
        c = np.array([float(c_i) for c_i in line[1:]], dtype=np.float32)

        return smiles, c



class SparseFP(Conditional):

    def __init__(self, fp_size=1024):
        self.fp_size = fp_size

    def __call__(self, line):
        record = line.strip('\n').strip('\r').split('\t')
        c = np.array([False, ]*self.fp_size, dtype=bool)
        smiles, on_bits = record[0], record[1:]
        if on_bits[0] != '':
            on_bits = [int(_i) for _i in on_bits]
            c[on_bits] = True
        return smiles, c


# from rdkit.Chem import QED
# import rdkit_contrib
#
# class QED_SA(Conditional):
#
#     def __init__(self):
#         pass
#
#     def __call__(self, line):
#         smiles = line.strip('\n').strip('\r')
#         mol = Chem.MolFromSmiles(smiles)
#         qed = QED.qed(smiles)
#         sa = rdkit_contrib.SA(mol)
#
#         c = np.array([qed, sa], dtype=np.float32)
#
#         return smiles, c

from rdkit import Chem
from rdkit.Chem import DataStructs

class ScaffoldFP(Conditional):

    def __init__(self, scaffolds):
        if isinstance(scaffolds, str):
            # input the directory of scaffold file
            with open(scaffolds) as f:
                scaffolds = [s.strip('\n').strip('\r') for s in f.readlines()]
        else:
            try:
                assert isinstance(scaffolds, list)
            except AssertionError:
                raise TypeError

        self.scaffolds = [Chem.MolFromSmiles(s) for s in scaffolds]
        self.scaffold_fps = [Chem.RDKFingerprint(s) for s in self.scaffolds]

    def get_on_bits(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        mol_fp = Chem.RDKFingerprint(mol)

        on_bits = []
        for i, s_fp_i in enumerate(self.scaffold_fps):
            if DataStructs.AllProbeBitsMatch(s_fp_i, mol_fp):
                if mol.HasSubstructMatch(self.scaffolds[i]):
                    on_bits.append(i)

        return on_bits

    def __call__(self, line):
        if isinstance(line, str):
            smiles = line.strip('\n').strip('\r')
        else:
            smiles = line
        c = np.array([False, ]*len(self.scaffolds), dtype=bool)
        c[self.get_on_bits(smiles)] = True
        return smiles, c
