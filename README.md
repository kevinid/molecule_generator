![abstract](https://github.com/kevinid/molecule_generator/blob/master/img/abstract.png?raw=true)



# Conditional Molecule Generator

This repository contains the source code and data sets for the graph based molecule generator discussed in the article "Multi-Objective De Novo Drug Design with Conditional Graph Generative Model" (https://arxiv.org/abs/1801.07299). 

Briefly speaking, we used conditional graph convolution to structure the generative model. The properties of output molecules can then be controlled using the conditional code.

## Requirements

This repo is built using Python 2.7, and utilizes the following packages:

- MXNet == 1.3.1
- RDKit == 2018.03.3
- Numpy
- Scikit-learn (for the predictive model)

To ease the installation process, please use the dockerfile environment defined in the `Dockerfile`.

## Quick start

### Project structure

- `train.py`: main training script.
- `mx_mg`: package for the molecule generative model:
  - `data`: packages for data processing workflows:
    - `conditionals.py`: callables used to generate the conditional codes for molecules
    - `data_struct.py`: defines atom types and bond types
    - `dataloaders.py` , `datasets.py` and `samplers.py`: data loading logics
    - `utils.py`: utility functions
  - `models`: library for graph generative models
    - `modules.py`: define modules (or blocks) such as graph convolution
    - `networks.py`: define networks (MolMP, MolRNN and CMolRNN)
    - `functions.py`: autograd.Function objects and operations
  - `builders.py`: utilities for building molecules using generative models
- `rdkit_contrib`: functions used to calculate QED and SAscore (for older version of rdkit)
- `example.ipynb`: tutorial

### Usage

To train the model, first unpack`datasets.tar.gz` (download [here](https://github.com/kevinid/molecule_generator/releases/download/1.0/datasets.tar.gz)) to the current directory, and call:
```shell
./train.py {molmp|molrnn|scaffold|prop|kinase} path/to/output
```
Where `{molmp|molrnn|scaffold|prop|kinase}` are model types, and `path/to/output` is the directory where you want to save the model's checkpoint file and log files. The following call:

```shell
./train.py {molmp|molrnn|scaffold|prop|kinase} -h
```

gives help for each model type.

## For any questions | problems | criticisms | ...

Please contact me. Email: [1210307427@pku.edu.cn](mailto:1210307427@pku.edu.cn) or [kevinid4g@gmail.com](mailto:kevinid4g@gmail.com)