#!/usr/bin/env python
import mxnet as mx
from mxnet import gluon, autograd, nd
import gc
import time
import os
import numpy as np
import json
import argparse
import sys

from mx_mg import models, data
from mx_mg.data import get_mol_spec


def _engine(file_name='datasets/ChEMBL.txt', ckpt_dir='ckpt/vanilla',
            is_full=False, num_folds=5, fold_id=0,
            batch_size=50, batch_size_test=100, num_workers=2, k=5, p=0.8,
            F_e=16, F_h=(32, 64, 128, 128, 256, 256), F_skip=256,
            F_c=(512, ), Fh_policy=128, activation='relu', N_rnn=None,
            gpu_ids=(0, 1, 2, 3), lr=1e-3, decay=0.01, decay_step=100, clip_grad=3.0, iterations=30000,
            summary_step=200):
    if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
        is_continuous = True
    else:
        is_continuous = False

    with open(file_name) as f:
        dataset = data.Lambda(f.readlines(), lambda _x:_x.strip('\n').strip('\r'))

    if is_full:
        db_train = dataset

        # get sampler and loader for training set
        sampler_train = data.BalancedSampler(cost=[len(l) for l in db_train], batch_size=batch_size)
        if N_rnn is not None:
            loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        else:
            loader_train = data.MolLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)

        it_train = iter(loader_train)

        loader_test, it_test = None, None
    else:
        # get dataset
        db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
        db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)

        # get sampler and loader for training set
        sampler_train = data.BalancedSampler(cost=[len(l) for l in db_train], batch_size=batch_size)
        if N_rnn is not None:
            loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        else:
            loader_train = data.MolLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)

        # get sampler and loader for test set
        sampler_test = data.BalancedSampler(cost=[len(l) for l in db_test], batch_size=batch_size_test)
        if N_rnn is not None:
            loader_test = data.MolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers, k=k, p=p)
        else:
            loader_test = data.MolLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers, k=k, p=p)

        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)

    if N_rnn is not None:
        if not is_continuous:
            configs = {'F_e': F_e,
                       'F_h': F_h,
                       'F_skip': F_skip,
                       'F_c': F_c,
                       'Fh_policy': Fh_policy,
                       'activation': activation,
                       'N_rnn': N_rnn}
            with open(os.path.join(ckpt_dir, 'configs.json'), 'w') as f:
                json.dump(configs, f)
        else:
            with open(os.path.join(ckpt_dir, 'configs.json')) as f:
                configs = json.load(f)

        model = models.VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)
    else:
        # build model
        if not is_continuous:
            configs = {'F_e': F_e,
                       'F_h': F_h,
                       'F_skip': F_skip,
                       'F_c': F_c,
                       'Fh_policy':Fh_policy,
                       'activation':activation}
            with open(os.path.join(ckpt_dir, 'configs.json'), 'w') as f:
                json.dump(configs, f)
        else:
            with open(os.path.join(ckpt_dir, 'configs.json')) as f:
                configs = json.load(f)

        model = models.VanillaMolGen(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

    ctx = [mx.gpu(i) for i in gpu_ids]
    if not is_continuous:
        model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    else:
        model.load_params(os.path.join(ckpt_dir, 'ckpt.params'), ctx=ctx)

    # construct optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    trainer = gluon.Trainer(model.collect_params(), opt)
    if is_continuous:
        trainer.load_states(os.path.join(ckpt_dir, 'trainer.status'))

    if not is_continuous:
        t0 = time.time()
        global_counter = 0
    else:
        with open(os.path.join(ckpt_dir, 'log.out')) as f:
            records = f.readlines()
            if records[-1] != 'Training finished\n':
                final_record = records[-1]
            else:
                final_record = records[-2]
        count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
        t0 = time.time() - t_final * 60
        global_counter = count

    with open(os.path.join(ckpt_dir, 'log.out'),
              mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            f.write('step\ttime(h)\tloss\tlr\n')
        while True:
            global_counter += 1

            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]

            # move to gpu
            if N_rnn is None:
                inputs = [data.MolLoader.from_numpy_to_tensor(input_i, j)
                          for j, input_i in zip(gpu_ids, inputs)]
            else:
                inputs = [data.MolRNNLoader.from_numpy_to_tensor(input_i, j)
                          for j, input_i in zip(gpu_ids, inputs)]

            with autograd.record():
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                loss = sum(loss) / len(gpu_ids)
                loss.backward()

            nd.waitall()
            gc.collect()

            trainer.step(batch_size=1)
            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))

            if global_counter % summary_step == 0:
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                else:
                    del loss, inputs
                    gc.collect()

                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]

                    with autograd.predict_mode():
                        # move to gpu
                        inputs = [data.MolLoader.from_numpy_to_tensor(input_i, j)
                                  for j, input_i in zip(gpu_ids, inputs)]
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())

                model.save_params(os.path.join(ckpt_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

                f.write('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0)/60, loss, trainer.learning_rate))
                f.flush()

                del loss, inputs
                gc.collect()

            if global_counter >= iterations:
                break

        # save before exit
        model.save_params(os.path.join(ckpt_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

        f.write('Training finished\n')


def _engine_cond(cond_type='scaffold', file_name='datasets/ChEMBL_scaffold.txt', num_scaffolds=734,
                 is_full=False,
                 ckpt_dir='ckpt/scaffold', num_folds=5, fold_id=0,
                 batch_size=50, batch_size_test=100, num_workers=2, k=5, p=0.8,
                 F_e=16, F_h=(32, 64, 128, 128, 256, 256), F_skip=256,
                 F_c=(512, ), Fh_policy=128, activation='relu', N_rnn=3,
                 gpu_ids=(0, 1, 2, 3), lr=1e-3, decay=0.015, decay_step=100, clip_grad=3.0,
                 iterations=30000, summary_step=200):
    if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
        is_continuous = True
    else:
        is_continuous = False

    if is_full:
        if cond_type != 'kinase':
            if cond_type == 'scaffold':
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type == 'prop':
                cond = data.Delimited()
                N_C = 2
            else:
                raise ValueError


            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get sampler and loader for training set
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in dataset], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(dataset, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)

            loader_test = []
        else:
            cond = data.Delimited()
            N_C = 2

            if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            db_train = data.Lambda(data.Filter(dataset,
                                               fn=lambda _x: not _filter(_x, fold_id)),
                                   fn=lambda _x: _x[:-2])
            db_test = data.Lambda(data.Filter(dataset,
                                              fn=lambda _x: _filter(_x, fold_id)),
                                  fn=lambda _x: _x[:-2])

            # get sampler and loader for test set
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers,
                                             k=k, p=p, conditional=cond, batch_size=batch_size_test)

            # get sampler and loader for training set
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers,
                                              k=k, p=p, conditional=cond, batch_size=batch_size)

        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)
    else:
        if cond_type != 'kinase':
            if cond_type=='scaffold':
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type=='prop':
                cond = data.Delimited()
                N_C = 2
            else:
                raise ValueError

            if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
            db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)

            # get sampler and loader for training set
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in db_train], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)

            # get sampler and loader for test set
            sampler_test = data.BalancedSampler(cost=[len(l.split('\t'[0])) for l in db_test], batch_size=batch_size_test)
            loader_test = data.CMolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers,
                                             k=k, p=p, conditional=cond)

        else:
            cond = data.Delimited()
            N_C = 2

            if all([os.path.isfile(os.path.join(ckpt_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            db_train = data.Lambda(data.Filter(dataset,
                                               fn=lambda _x: not _filter(_x, fold_id)),
                                   fn=lambda _x: _x[:-2])
            db_test = data.Lambda(data.Filter(dataset,
                                              fn=lambda _x: _filter(_x, fold_id)),
                                  fn=lambda _x: _x[:-2])

            # get sampler and loader for training set
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers,
                                              k=k, p=p, conditional=cond, batch_size=batch_size)

            # get sampler and loader for test set
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers,
                                             k=k, p=p, conditional=cond, batch_size=batch_size_test)

        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)

    # build model
    if not is_continuous:
        configs = {'N_C': N_C,
                   'F_e': F_e,
                   'F_h': F_h,
                   'F_skip': F_skip,
                   'F_c': F_c,
                   'Fh_policy': Fh_policy,
                   'activation': activation,
                   'rename': True,
                   'N_rnn': N_rnn}
        with open(os.path.join(ckpt_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
    else:
        with open(os.path.join(ckpt_dir, 'configs.json')) as f:
            configs = json.load(f)

    model = models.CVanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

    ctx = [mx.gpu(i) for i in gpu_ids]
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    if not is_continuous:
        if cond_type == 'kinase':
            model.load_params(os.path.join(ckpt_dir, 'ckpt.params.bk'), ctx=ctx, allow_missing=True)
    else:
        model.load_params(os.path.join(ckpt_dir, 'ckpt.params'), ctx=ctx)

    # construct optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    trainer = gluon.Trainer(model.collect_params(), opt)
    if is_continuous:
        trainer.load_states(os.path.join(ckpt_dir, 'trainer.status'))

    if not is_continuous:
        t0 = time.time()
        global_counter = 0
    else:
        with open(os.path.join(ckpt_dir, 'log.out')) as f:
            records = f.readlines()
            if records[-1] != 'Training finished\n':
                final_record = records[-1]
            else:
                final_record = records[-2]
        count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
        t0 = time.time() - t_final * 60
        global_counter = count

    with open(os.path.join(ckpt_dir, 'log.out'),
              mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            f.write('step\ttime(h)\tloss\tlr\n')
        while True:
            global_counter += 1

            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]

            # move to gpu
            inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j)
                      for j, input_i in zip(gpu_ids, inputs)]

            with autograd.record():
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                loss = sum(loss) / len(gpu_ids)
                loss.backward()

            nd.waitall()
            gc.collect()

            trainer.step(batch_size=1)

            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))

            if global_counter % summary_step == 0:
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                else:
                    del loss, inputs
                    gc.collect()

                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]

                    with autograd.predict_mode():
                        # move to gpu
                        inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j)
                                  for j, input_i in zip(gpu_ids, inputs)]
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())

                model.save_params(os.path.join(ckpt_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

                f.write('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0) / 60, loss,
                                                  trainer.learning_rate))
                f.flush()

                del loss, inputs
                gc.collect()

            if global_counter >= iterations:
                break

        # save before exit
        model.save_params(os.path.join(ckpt_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(ckpt_dir, 'trainer.status'))

        f.write('Training finished\n')

# parsers
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('ckpt_dir', action='store', default='.',
                           help='Location where the training results and model will be stored, '
                                'default to the current directory')

parent_parser.add_argument('--full', dest='is_full', action='store_true',
                           help='Train using the full dataset')
parent_parser.add_argument('--num-folds', dest='num_folds', action='store', type=int, default=5,
                           help='Specify the number of folds used in cross validation, default to 5')
parent_parser.add_argument('--fold-id', dest='fold_id', action='store', type=int, default=0,
                           help='Specify which fold is used as test set, default to 0')

# Parameters for data loader
parent_parser.add_argument('--batch-size', dest='batch_size', action='store', type=int, default=50)
parent_parser.add_argument('--batch-size-test', dest='batch_size_test', action='store', type=int, default=100,
                           help='Mini-batch size for evaluation during training, default to 100')
parent_parser.add_argument('--num-workers', dest='num_workers', action='store', type=int, default=1,
                           help='Number of worker for data processing, default to 1')
parent_parser.add_argument('--k', dest='k', action='store', type=int, default=5,
                           help='Number of decoding route used, default to 5')
parent_parser.add_argument('--alpha', dest='p', action='store', type=float, default=0.8,
                           help='Parameter controlling the randomness of importance sampling, '
                                'default to 0.8')

# Model architecture
parent_parser.add_argument('--embedding-size', dest='F_e', action='store', type=int, default=16,
                           help='Size of the initial atom embedding, default to 16')
parent_parser.add_argument('--gc-size', dest='F_h', action='store',
                           type=int, nargs='+', default=[32, 64, 128, 128, 256, 256],
                           help='Hidden size for graph convolution layers, should be provided as a list')
parent_parser.add_argument('--skip-size', dest='F_skip', action='store', type=int, default=256,
                           help='Size of skip connection layer, default ot 256')
parent_parser.add_argument('--fc-size', dest='F_c', action='store',
                           type=int, nargs='+', default=[512, ],
                           help='The hidden sizes of fully connected layers after graph convolution, '
                                'should be provided as a list')
parent_parser.add_argument('--policy-size', dest='Fh_policy', action='store', type=int, default=128,
                           help='Hidden size for policy layer, default to 0.8')
parent_parser.add_argument('--activation', dest='activation', action='store', default='relu',
                           choices=['relu', 'tanh'],
                           help='The type of activation function used, default to relu')


# Training details
parent_parser.add_argument('--gpu', dest='gpu_ids', action='store',
                           type=int, nargs='+', default=[0, ],
                           help='GPUs used in the training, default to the first GPU')
parent_parser.add_argument('--learning-rate', dest='lr', action='store', type=float, default=1e-3,
                           help='The initial learning rate of Adam optimizer, default to 1e-3')
parent_parser.add_argument('--decay', dest='decay', action='store', type=float, default=0.01,
                           help='The rate of learning rate decay, default to 0.01')
parent_parser.add_argument('--decay-step', dest='decay_step', action='store', type=int, default=100,
                           help='Perform learning rate decay in every \"--decay-step\" steps, default to 100')
parent_parser.add_argument('--clip-grad', dest='clip_grad', action='store', type=float, default=3.0)
parent_parser.add_argument('--iterations', dest='iterations', action='store', type=int, default=30000,
                           help='Number of iterations to perform during the training, default to 30,000 iterations')
parent_parser.add_argument('--summary-step', dest='summary_step', action='store', type=int, default=200,
                           help='Output performance metrics and model checkpoints for every \"--summary-step\" steps, '
                                'default to 200 steps')

# build the main parser
parser = argparse.ArgumentParser(description='Script for model training')
sub_parsers = parser.add_subparsers()

# parser for MolMP
mol_mp_parser = sub_parsers.add_parser('molmp', description='Train unconditional MolMP', parents=[parent_parser, ])
mol_mp_parser.add_argument('--file-name', action='store', dest='file_name', default='datasets/ChEMBL.txt',
                           help='Location of the training dataset, default to datasets/ChEMBL.txt')

# parser for MolRNN
mol_rnn_parser = sub_parsers.add_parser('molrnn', description='Train unconditional MolRNN', parents=[parent_parser, ])
mol_rnn_parser.add_argument('--file-name', action='store', dest='file_name', default='datasets/ChEMBL.txt',
                            help='Location of the training dataset, default to datasets/ChEMBL.txt')
mol_rnn_parser.add_argument('--num-gru-layers', action='store', dest='N_rnn', type=int, default=3,
                            help='Number of layers used in GRUs, default to 3')

# parser for scaffold based training
scaffold_parser = sub_parsers.add_parser('scaffold', description='Train scaffold based conditional generator',
                                         parents=[parent_parser, ])
scaffold_parser.add_argument('--file-name', action='store', dest='file_name', default='datasets/ChEMBL_scaffold.txt',
                             help='Location of the training dataset, default to datasets/ChEMBL_scaffold.txt')
scaffold_parser.add_argument('--num-gru-layers', action='store', dest='N_rnn', type=int, default=3,
                             help='Number of layers used in GRUs, default to 3')

# parser for property based training
prop_parser = sub_parsers.add_parser('prop', description='Train property based conditional generator',
                                         parents=[parent_parser, ])
prop_parser.add_argument('--file-name', action='store', dest='file_name', default='datasets/ChEMBL_prop.txt',
                         help='Location of the training dataset, default to datasets/ChEMBL_prop.txt')
prop_parser.add_argument('--num-gru-layers', action='store', dest='N_rnn', type=int, default=3,
                         help='Number of layers used in GRUs, default to 3')

# parser for kinase based training
kinase_parser = sub_parsers.add_parser('kinase', description='Train conditional generator based on GSK-3b and JNK3',
                                         parents=[parent_parser, ])
kinase_parser.add_argument('--file-name', action='store', dest='file_name', default='datasets/ChEMBL_k.txt',
                         help='Location of the training dataset, default to datasets/ChEMBL_kinase.txt')
kinase_parser.add_argument('--num-gru-layers', action='store', dest='N_rnn', type=int, default=3,
                         help='Number of layers used in GRUs, default to 3')

if __name__ == '__main__':
    params = vars(parser.parse_args())
    if sys.argv[1] in ['molmp', 'molrnn']:
        _engine(**params)
    else:
        params['cond_type'] = sys.argv[1]
        _engine_cond(**params)
