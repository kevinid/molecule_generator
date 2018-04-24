import random
from scipy import sparse
from rdkit import Chem
import networkx as nx
import numpy as np
from multiprocessing import Queue, Process
from threading import Thread
import gzip

import meta


def from_txt(file_name, filter_fn=lambda _x:False,
             preprocess_fn=lambda _x:_x.strip('\n').strip('\r'),
             num_skips=0, is_train=False, num_epoch=1, do_shuffle=False):
    if '.gzip' in file_name:
        def _g():
            with gzip.open(file_name, 'rt') as f_:
                for i, line_ in enumerate(f_):
                    if num_skips is not None:
                        if is_train:
                            if i >= num_skips:
                                break
                        else:
                            if i < num_skips:
                                continue
                    if line_ != '' and not filter_fn(line_):
                        yield preprocess_fn(line_) if preprocess_fn is not None else line_
        epoch_count = 0
        while True:
            if num_epoch is not None:
                if epoch_count >= num_epoch:
                    break
            for line in _g():
                yield line
            epoch_count += 1
    else:
        # for filter_fn: return True if line is to be filtered
        with open(file_name) as f:
            lines = [preprocess_fn(line) if preprocess_fn is not None else line
                     for line in f if line != '' and not filter_fn(line)]
        if num_skips is not None:
            if not is_train:
                lines = lines[num_skips:]
            else:
                lines = lines[:num_skips]

        epoch_count = 0
        while True:
            if num_epoch is not None:
                if epoch_count >= num_epoch:
                    break
            if do_shuffle:
                random.shuffle(lines)
            for line in lines:
                yield line
            epoch_count += 1


def get_graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # build graph
    atom_types, atom_ranks, bonds, bond_types = [], [], [], []
    for a, r in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol)):
        atom_types.append(meta.atom_to_index(a))
        atom_ranks.append(r)
    for b in mol.GetBonds():
        idx_1, idx_2, bt = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), meta.bond_to_index(b)
        bonds.append([idx_1, idx_2])
        bond_types.append(bt)

    # build nx graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_types)))
    graph.add_edges_from(bonds)

    return graph, atom_types, atom_ranks, bonds, bond_types


def from_smiles_to_graph(smiles_list):
    graph_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        # build graph
        atom_types, bonds, bond_types = [], [], []
        for a in mol.GetAtoms():
            atom_types.append(meta.atom_to_index(a))
        for b in mol.GetBonds():
            idx_1, idx_2, bt = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), meta.bond_to_index(b)
            bonds.append([idx_1, idx_2])
            bond_types.append(bt)

        X_0 = np.array(atom_types, dtype=np.int32)
        A_0 = np.concatenate([np.array(bonds, dtype=np.int32),
                              np.array(bond_types, dtype=np.int32)[:, np.newaxis]],
                             axis=1)
        graph_list.append([X_0, A_0])
    return graph_list


def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None, p=0.9, log_p=0.0):
    if current_node is None:
        next_nodes = range(len(atom_ranks))
        step_ids = [-1, ] * len(next_nodes)
        next_node_ranks = atom_ranks
    else:
        next_nodes = graph.neighbors(current_node)  # get neighbor nodes
        next_nodes = [n for n in next_nodes if step_ids[n] < 0] # filter visited nodes
        next_node_ranks = [atom_ranks[n] for n in next_nodes] # get ranks for neighbors
    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks), key=lambda _x:_x[1])] # sort by rank

    # iterate through neighbors
    while len(next_nodes) > 0:
        if len(next_nodes)==1:
            next_node = next_nodes[0]
        elif random.random() >= (1 - p):
            next_node = next_nodes[0]
            log_p += np.log(p)
        else:
            next_node = next_nodes[random.randint(1, len(next_nodes) - 1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))
        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph(graph, atom_ranks, next_node, step_ids, p, log_p)
        next_nodes = [n for n in next_nodes if step_ids[n] < 0] # filter visited nodes

    return step_ids, log_p


def single_reorder(X_0, A_0, step_ids):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)

    step_ids = np.array(step_ids, dtype=np.int32)

    # sort by step_ids
    sorted_ids = np.argsort(step_ids)
    X_0 = X_0[sorted_ids]
    A_0[:, 0], A_0[:, 1] = step_ids[A_0[:, 0]], step_ids[A_0[:, 1]]
    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    A_0 = A_0[np.lexsort([-min_b, max_b]), :]

    # separate append and connect
    max_b, min_b = np.amax(A_0[:, :2], axis=1), np.amin(A_0[:, :2], axis=1)
    is_append = np.concatenate([np.array([True]), max_b[1:] > max_b[:-1]])
    A_0 = np.concatenate([np.where(is_append[:, np.newaxis],
                                 np.stack([min_b, max_b], axis=1),
                                 np.stack([max_b, min_b], axis=1)),
                        A_0[:, -1:]], axis=1)

    return X_0, A_0


def single_expand(X_0, A_0):
    X_0, A_0 = np.copy(X_0), np.copy(A_0)

    # expand X
    is_append_iter = np.less(A_0[:, 0], A_0[:, 1]).astype(np.int32)
    NX = np.cumsum(np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1))
    shift = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    X_index = np.arange(NX.sum(), dtype=np.int32) - np.repeat(shift, NX)
    X = X_0[X_index]

    # expand A
    _, A_index = np.tril_indices(A_0.shape[0])
    A = A_0[A_index, :]
    NA = np.arange(A_0.shape[0] + 1)

    # get action
    # action_type, atom_type, bond_type, append_pos, connect_pos
    action_type = 1 - is_append_iter
    atom_type = np.where(action_type == 0, X_0[A_0[:, 1]], 0)
    bond_type = A_0[:, 2]
    append_pos = np.where(action_type == 0, A_0[:, 0], 0)
    connect_pos = np.where(action_type == 1, A_0[:, 1], 0)
    actions = np.stack([action_type, atom_type, bond_type, append_pos, connect_pos],
                       axis=1)
    last_action = [[2, 0, 0, 0, 0]]
    actions = np.append(actions, last_action, axis=0)

    action_0 = np.array([X_0[0]], dtype=np.int32)

    # }}}

    # {{{ Get mask
    last_atom_index = shift + NX - 1
    last_atom_mask = np.zeros_like(X)
    last_atom_mask[last_atom_index] = np.where(
        np.pad(is_append_iter, [[1, 0]], mode='constant', constant_values=1) == 1,
        np.ones_like(last_atom_index),
        np.ones_like(last_atom_index) * 2)
    # }}}

    return action_0, X, NX, A, NA, actions, last_atom_mask


def get_d(A, X):
    _to_sparse = lambda _A, _X: sparse.coo_matrix((np.ones([_A.shape[0] * 2], dtype=np.int32),
                                                   (np.concatenate([_A[:, 0], _A[:, 1]], axis=0),
                                                    np.concatenate([_A[:, 1], _A[:, 0]], axis=0))),
                                                  shape=[_X.shape[0], ] * 2)
    A_sparse = _to_sparse(A, X)

    d2 = A_sparse * A_sparse
    d3 = d2 * A_sparse

    # get D_2
    D_2 = np.stack(d2.nonzero(), axis=1)
    D_2 = D_2[D_2[:, 0] < D_2[:, 1], :]

    # get D_3
    D_3 = np.stack(d3.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]

    # remove D_1 elements from D_3
    D_3_sparse = _to_sparse(D_3, X)
    D_3_sparse = D_3_sparse - D_3_sparse.multiply(A_sparse)
    D_3 = np.stack(D_3_sparse.nonzero(), axis=1)
    D_3 = D_3[D_3[:, 0] < D_3[:, 1], :]

    return D_2, D_3


def merge_single_0(X_0, A_0, NX_0, NA_0):
    # shift_ids
    cumsum = np.cumsum(np.pad(NX_0, [[1, 0]], mode='constant')[:-1])
    A_0[:, :2] += np.stack([np.repeat(cumsum, NA_0), ] * 2, axis=1)

    # get D
    D_0_2, D_0_3 = get_d(A_0, X_0)

    # split A
    A_split = []
    for i in range(meta.NUM_BOND_TYPES):
        A_i = A_0[A_0[:, 2] == i, :2]
        A_split.append(A_i)
    A_split.extend([D_0_2, D_0_3])
    A_0 = A_split

    # NX_rep
    NX_rep_0 = np.repeat(np.arange(NX_0.shape[0]), NX_0)

    return X_0, A_0, NX_0, NX_rep_0


def merge_single(X, A,
                 NX, NA,
                 mol_ids, rep_ids, iw_ids,
                 action_0, actions,
                 last_append_mask,
                 log_p):
    X, A, NX, NX_rep = merge_single_0(X, A, NX, NA)
    cumsum = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')[:-1])
    actions[:, -2] += cumsum * (actions[:, 0] == 0)
    actions[:, -1] += cumsum * (actions[:, 0] == 1)
    mol_ids_rep = np.repeat(mol_ids, NX)
    rep_ids_rep = np.repeat(rep_ids, NX)

    return X, A,\
           mol_ids_rep, rep_ids_rep, iw_ids,\
           last_append_mask,\
           NX, NX_rep,\
           action_0, actions, \
           log_p

def process_single(smiles, k, p):
    graph, atom_types, atom_ranks, bonds, bond_types = get_graph_from_smiles(smiles)

    # original
    X_0 = np.array(atom_types, dtype=np.int32)
    A_0 = np.concatenate([np.array(bonds, dtype=np.int32),
                          np.array(bond_types, dtype=np.int32)[:, np.newaxis]],
                         axis=1)

    X, A = [], []
    NX, NA = [], []
    mol_ids, rep_ids, iw_ids = [], [], []
    action_0, actions = [], []
    last_append_mask = []
    log_p = []

    # random sampling decoding route
    for i in range(k):
        step_ids_i, log_p_i = traverse_graph(graph, atom_ranks, p=p)
        X_i, A_i = single_reorder(X_0, A_0, step_ids_i)
        action_0_i, X_i, NX_i, A_i, NA_i, actions_i, last_atom_mask_i = single_expand(X_i, A_i)

        # appends
        X.append(X_i)
        A.append(A_i)
        NX.append(NX_i)
        NA.append(NA_i)
        action_0.append(action_0_i)
        actions.append(actions_i)
        last_append_mask.append(last_atom_mask_i)

        mol_ids.append(np.zeros_like(NX_i, dtype=np.int32))
        rep_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)
        iw_ids.append(np.ones_like(NX_i, dtype=np.int32) * i)

        log_p.append(log_p_i)

    # concatenate
    X = np.concatenate(X, axis=0)
    A = np.concatenate(A, axis = 0)
    NX = np.concatenate(NX, axis = 0)
    NA = np.concatenate(NA, axis = 0)
    action_0 = np.concatenate(action_0, axis = 0)
    actions = np.concatenate(actions, axis = 0)
    last_append_mask = np.concatenate(last_append_mask, axis = 0)
    mol_ids = np.concatenate(mol_ids, axis = 0)
    rep_ids = np.concatenate(rep_ids, axis = 0)
    iw_ids = np.concatenate(iw_ids, axis = 0)
    log_p = np.array(log_p, dtype=np.float32)

    return X, A, NX, NA, mol_ids, rep_ids, iw_ids, action_0, actions, last_append_mask, log_p


def from_smiles_to_batch(input_g, batch_size=25,
                         queue_in_size=10, queue_out_size=10, num_pool=10,
                         k=10, p=0.9, conditional=False, is_train=False):
    queue_in = Queue(queue_in_size)
    queue_out = Queue(queue_out_size)

    def _target_w():
        # names = X, A,
        #         NX, NA,
        #         mol_ids, rep_ids, iw_ids,
        #         action_0, actions,
        #         last_append_mask,
        #         log_p

        shapes = [[0], [0, 3],
                  [0], [0],
                  [0], [0], [0],
                  [0], [0, 5],
                  [0],
                  [0]]
        dtypes = [np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32, np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32,
                  np.float32]

        _build = lambda : [np.zeros(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
        _append = lambda _r0, _r1: [np.concatenate([__r0, __r1], axis=0)
                                   for __r0, __r1 in zip(_r0, _r1)]

        X, A, \
        NX, NA, \
        mol_ids, rep_ids, iw_ids, \
        action_0, actions, \
        last_append_mask , \
        log_p = _build()

        if conditional:
            c = []
        else:
            c = None

        counter = 0
        while True:
            record_in = queue_in.get()
            if record_in == 'end':
                queue_out.put('end')
                break
            else:
                if conditional:
                    record_in = record_in.split('\t')
                    smiles = record_in[0]
                    c_i = [float(val) for val in record_in[1:]]
                    c.append(c_i)
                else:
                    smiles = record_in

                X_i, A_i, \
                NX_i, NA_i, \
                mol_ids_i, rep_ids_i, iw_ids_i, \
                action_0_i, actions_i, \
                last_append_mask_i, log_p_i = process_single(smiles, k, p)

                if counter != 0:
                    mol_ids_i += mol_ids[-1] + 1
                    iw_ids_i += iw_ids[-1] + 1


                X, A, \
                NX, NA, \
                mol_ids, rep_ids, iw_ids, \
                action_0, actions, \
                last_append_mask , \
                log_p = _append([X, A,
                                 NX, NA,
                                 mol_ids, rep_ids, iw_ids,
                                 action_0, actions,
                                 last_append_mask ,
                                 log_p],
                                [X_i, A_i,
                                 NX_i, NA_i,
                                 mol_ids_i, rep_ids_i, iw_ids_i,
                                 action_0_i, actions_i,
                                 last_append_mask_i,
                                 log_p_i])
                counter += 1
                if counter >= batch_size:
                    X, A, \
                    mol_ids_rep, rep_ids_rep, iw_ids, \
                    last_append_mask, \
                    NX, NX_rep, \
                    action_0, actions, \
                    log_p = merge_single(X, A,
                                         NX, NA,
                                         mol_ids, rep_ids, iw_ids,
                                         action_0, actions,
                                         last_append_mask,
                                         log_p)

                    result_out = [X, A,
                                  mol_ids_rep, rep_ids_rep, iw_ids,
                                  last_append_mask,
                                  NX, NX_rep,
                                  action_0, actions,
                                  log_p]

                    if c is not None:
                        c = np.array(c, dtype=np.float32)
                        result_out.append(c)
                        c = []

                    queue_out.put(result_out)
                    X, A, \
                    NX, NA, \
                    mol_ids, rep_ids, iw_ids, \
                    action_0, actions, \
                    last_append_mask, \
                    log_p = _build()
                    counter = 0

    def _target_g():
        for smiles in input_g:
            queue_in.put(smiles)
        if num_pool is None:
            queue_in.put('end')
        else:
            for _ in range(num_pool):
                queue_in.put('end')

    t_g = Thread(target=_target_g)
    t_g.start()

    if num_pool is None:
        t_w = Thread(target=_target_w)
        t_w.start()
        while True:
            record = queue_out.get()
            if isinstance(record,str) and record == 'end':
                break
            else:
                yield record
    else:
        t_w = [Process(target=_target_w) for _ in range(num_pool)]
        for t in t_w:
            t.start()
        end_counter = 0
        while end_counter < num_pool:
            record = queue_out.get()
            if isinstance(record, str) and record == 'end':
                end_counter += 1
            else:
                yield record

GENERATORS = {
    'from_smiles_to_batch':from_smiles_to_batch
}

def kinase_down_sample(input_g, keep_rate=0.1):
    for record in input_g:
        _, c_1, c_2 = record.split('\t')
        c_1, c_2 = int(c_1), int(c_2)
        if c_1 == 1 or c_2 == 1:
            yield record
        else:
            if random.random() < keep_rate:
                yield record
            else:
                continue

SAMPLERS = {
    'kinase_sampler':kinase_down_sample
}


# noinspection PyArgumentList
def from_graph_to_mol(X, A, sanitize=True):
    try:
        mol = Chem.RWMol(Chem.Mol())

        X, A = X.tolist(), A.tolist()
        for i, atom_type in enumerate(X):
            mol.AddAtom(meta.index_to_atom(atom_type))

        for atom_id1, atom_id2, bond_type in A:
            meta.index_to_bond(mol, atom_id1, atom_id2, bond_type)
    except:
        return None

    if sanitize:
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    else:
        return mol

