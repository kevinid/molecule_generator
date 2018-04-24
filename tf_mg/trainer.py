import tensorflow as tf
import os
import json
import gc

import network
import data


def multi_gpu_trainer(model_dir, job_name, task_index):

    with open(os.path.join(model_dir, 'model.json')) as f:
        model_params = json.load(f)

    with open(os.path.join(model_dir, 'trainer.json')) as f:
        trainer_params = json.load(f)


    cluster_dict = {'ps': ['localhost:{}'.format(trainer_params['port_base'])],
                    'worker': ['localhost:{}'.format(trainer_params['port_base'] + i + 1)
                               for i in range(trainer_params['num_gpu'])],
                    'generator': ['localhost:{}'.format(trainer_params['port_base'] +
                                                        trainer_params['num_gpu'] + i + 1)
                                  for i in range(2)]}

    cluster = tf.train.ClusterSpec(cluster_dict)

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)

    with tf.device('/job:ps/task:0/cpu:0'):
        mdl_class = network.NETWORKS[trainer_params['model_name']]
        mdl = mdl_class(model_params) # type: network.Network

    with tf.device('/job:generator/replica:0/task:0/cpu:0'):
        with tf.name_scope('train_data'):
            input_desc_train = mdl.get_input_desc('train') # type: network.InputDescList
            train_inputs, train_enqueue, train_dequeue = input_desc_train.get_queue(3 * trainer_params['num_gpu'],
                                                                                    trainer_params['num_gpu'],
                                                                                    shared_name='train_queue')

    with tf.device('/job:generator/replica:0/task:1/cpu:0'):
        with tf.name_scope('val_data'):
            input_desc_val = mdl.get_input_desc('summarize') # type: network.InputDescList
            val_inputs, val_enqueue, val_dequeue = input_desc_val.get_queue(1, 1)

    with tf.Session(server.target,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if job_name == 'ps':
            server.join()

        elif job_name == 'generator':
            sess.graph.finalize()
            if task_index == 0:
                g_txt = data.from_txt(trainer_params['dataset']['file_name'],
                                      num_epoch=trainer_params['dataset']['num_epoch'],
                                      num_skips=trainer_params['dataset']['split'],
                                      is_train=True, do_shuffle=True)
                if trainer_params['dataset']['sampler'] is not None:
                    g_txt = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt,
                                                                    **trainer_params['dataset']['sampler_config'])
                g = data.GENERATORS[trainer_params['dataset']['generator']](g_txt,
                                                                 batch_size=trainer_params['dataset']['batch_size'],
                                                                is_train=True,
                                                                 num_pool=trainer_params['dataset']['num_pool'],
                                                                 **trainer_params['dataset']['kwargs'])
                for i, record in enumerate(g):
                    is_match, msg = input_desc_train.check_inputs(record)
                    if not is_match:
                        raise Exception('Input do not match:' + msg)
                    feed_dict = network.get_feed_dict(train_inputs, record)
                    sess.run(train_enqueue, feed_dict)
                    if (i + 1) % 100 == 0:
                        print 'Generated {} mini-batches'.format(i + 1)

                print 'Data generation is completed, please kill the training process manually'

                server.join()

            elif task_index == 1:
                g_txt = data.from_txt(trainer_params['dataset']['file_name'],
                                      num_epoch=trainer_params['dataset']['num_epoch'],
                                      num_skips=trainer_params['dataset']['split'],
                                      is_train=False, do_shuffle=True)
                if trainer_params['dataset']['sampler'] is not None:
                    g_txt = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt,
                                                                    **trainer_params['dataset']['sampler_config'])
                g = data.GENERATORS[trainer_params['dataset']['generator']](g_txt,
                                                                            is_train=False,
                                                                 batch_size=trainer_params['dataset']['batch_size'],
                                                                 num_pool=None,
                                                                 **trainer_params['dataset']['kwargs'])
                for record in g:
                    is_match, msg = input_desc_val.check_inputs(record)
                    if not is_match:
                        raise Exception('Input do not match: ' + msg)
                    feed_dict = network.get_feed_dict(val_inputs, record)
                    sess.run(val_enqueue, feed_dict)
            else:
                raise Exception('Unrecognized task index')

        elif job_name == 'worker':
            with tf.device('/job:worker/replica:0/task:{}/gpu:0'.format(task_index)):
                # training
                step_op, global_step, total_loss = mdl('train', train_dequeue[task_index])
                # testing
                summary_op, global_step = mdl('summarize', val_dequeue[0])

            # checking initialization
            if task_index == 0:
                saver = tf.train.Saver()
                sw = tf.summary.FileWriter(model_dir)
                if os.path.isfile(os.path.join(model_dir, 'ckpt.meta')):
                    saver.restore(sess, os.path.join(model_dir, 'ckpt'))
                else:
                    sess.run(tf.global_variables_initializer())
                    if len(sess.run(tf.report_uninitialized_variables())) > 0:
                        raise Exception('uninitialized variable exist !')
            else:
                check_op = tf.report_uninitialized_variables()
                while len(sess.run(check_op)) > 0:
                    # wait for all
                    continue

            sess.graph.finalize()

            local_step = 0
            try:
                while True:
                    if local_step == 0:
                        _, current_step, loss = sess.run([step_op, global_step, total_loss])
                    else:
                        _, current_step, loss = sess.run([step_op, global_step, total_loss],
                                                  options=tf.RunOptions(timeout_in_ms=10000))
                    local_step += 1
                    if local_step % trainer_params['summarize_step'] == 0 and task_index == 0:
                        # perform summarize
                        summary, current_step = sess.run([summary_op, global_step])
                        sw.add_summary(summary, current_step)
                        saver.save(sess, os.path.join(model_dir, 'ckpt'))
                        sw.flush()
            except tf.errors.DeadlineExceededError:
                if task_index == 0:
                    summary, current_step = sess.run([summary_op, global_step])
                    sw.add_summary(summary, current_step)
                    saver.save(sess, os.path.join(model_dir, 'ckpt'))
                    sw.flush()
                    print 'checkpoint saved !'
                server.join()
        else:
            raise Exception('Unaccepted job name')

def multi_gpu_trainer_full(model_dir, job_name, task_index):

    with open(os.path.join(model_dir, 'model.json')) as f:
        model_params = json.load(f)

    with open(os.path.join(model_dir, 'trainer.json')) as f:
        trainer_params = json.load(f)

    cluster_dict = {'ps': ['localhost:{}'.format(trainer_params['port_base'])],
                    'worker': ['localhost:{}'.format(trainer_params['port_base'] + i + 1)
                               for i in range(trainer_params['num_gpu'])],
                    'generator': ['localhost:{}'.format(trainer_params['port_base'] +
                                                        trainer_params['num_gpu'] + 1)]}

    cluster = tf.train.ClusterSpec(cluster_dict)

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)

    with tf.device('/job:ps/task:0/cpu:0'):
        mdl_class = network.NETWORKS[trainer_params['model_name']]
        mdl = mdl_class(model_params) # type: network.Network

    with tf.device('/job:generator/replica:0/task:0/cpu:0'):
        with tf.name_scope('train_data'):
            input_desc_train = mdl.get_input_desc('train') # type: network.InputDescList
            train_inputs, train_enqueue, train_dequeue = input_desc_train.get_queue(3 * trainer_params['num_gpu'],
                                                                                    trainer_params['num_gpu'],
                                                                                    shared_name='train_queue')


    with tf.Session(server.target,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if job_name == 'ps':
            server.join()

        elif job_name == 'generator':
            sess.graph.finalize()
            g_txt = data.from_txt(trainer_params['dataset']['file_name'],
                                  num_epoch=trainer_params['dataset']['num_epoch'],
                                  num_skips=None,
                                  is_train=True, do_shuffle=True)
            if trainer_params['dataset']['sampler'] is not None:
                g_txt = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt,
                                                                **trainer_params['dataset']['sampler_config'])
            g = data.GENERATORS[trainer_params['dataset']['generator']](g_txt,
                                                                        batch_size=trainer_params['dataset']['batch_size'],
                                                                        is_train=True,
                                                                        num_pool=trainer_params['dataset']['num_pool'],
                                                                        **trainer_params['dataset']['kwargs'])
            for i, record in enumerate(g):
                is_match, msg = input_desc_train.check_inputs(record)
                if not is_match:
                    raise Exception('Input do not match:' + msg)
                feed_dict = network.get_feed_dict(train_inputs, record)
                sess.run(train_enqueue, feed_dict)
                if (i + 1) % 100 == 0:
                    print 'Generated {} mini-batches'.format(i + 1)

            print 'Data generation is completed, please kill the training process manually'

            server.join()


        elif job_name == 'worker':
            with tf.device('/job:worker/replica:0/task:{}/gpu:0'.format(task_index)):
                step_op, global_step, total_loss = mdl('train', train_dequeue[task_index])

            # checking initialization
            if task_index == 0:
                saver = tf.train.Saver()
                if os.path.isfile(os.path.join(model_dir, 'ckpt.meta')):
                    saver.restore(sess, os.path.join(model_dir, 'ckpt'))
                else:
                    sess.run(tf.global_variables_initializer())
                    if len(sess.run(tf.report_uninitialized_variables())) > 0:
                        raise Exception('uninitialized variable exist !')
            else:
                check_op = tf.report_uninitialized_variables()
                while len(sess.run(check_op)) > 0:
                    # wait for all
                    continue

            sess.graph.finalize()

            local_step = 0
            try:
                while True:
                    if local_step == 0:
                        _, current_step, loss = sess.run([step_op, global_step, total_loss])
                    else:
                        _, current_step, loss = sess.run([step_op, global_step, total_loss],
                                                  options=tf.RunOptions(timeout_in_ms=10000))
                    local_step += 1
                    if local_step % trainer_params['summarize_step'] == 0 and task_index == 0:
                        # perform summarize
                        saver.save(sess, os.path.join(model_dir, 'ckpt'))
                        msg = 'Reached global step: {}, training loss: {}'.format(current_step, loss)
                        with open('output.log', 'a') as f:
                            f.write(msg + '\n')
            except tf.errors.DeadlineExceededError:
                if task_index == 0:
                    saver.save(sess, os.path.join(model_dir, 'ckpt'))
                    with open('output.log', 'a') as f:
                        f.write('Finished training'+ '\n')
                server.join()
        else:
            raise Exception('Unaccepted job name')

def single_gpu_trainer(model_dir):
    with open(os.path.join(model_dir, 'model.json')) as f:
        model_params = json.load(f)

    with open(os.path.join(model_dir, 'trainer.json')) as f:
        trainer_params = json.load(f)

    g_txt_train = data.from_txt(trainer_params['dataset']['file_name'],
                                num_epoch=trainer_params['dataset']['num_epoch'],
                                num_skips=trainer_params['dataset']['split'],
                                is_train=True, do_shuffle=True)
    if trainer_params['dataset']['sampler'] is not None:
        g_txt_train = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt_train,
                                                                          **trainer_params['dataset']['sampler_config'])
    g_train = data.GENERATORS[trainer_params['dataset']['generator']](g_txt_train,
                                                                      batch_size=trainer_params['dataset']['batch_size'],
                                                                      num_pool=trainer_params['dataset']['num_pool'],
                                                                      **trainer_params['dataset']['kwargs'])

    g_txt_val = data.from_txt(trainer_params['dataset']['file_name'],
                              num_epoch=None,
                              num_skips=trainer_params['dataset']['split'],
                              is_train=False, do_shuffle=True)
    if trainer_params['dataset']['sampler'] is not None:
        g_txt_val = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt_val,
                                                                        **trainer_params['dataset']['sampler_config'])
    g_val = data.GENERATORS[trainer_params['dataset']['generator']](g_txt_val,
                                                                    batch_size=trainer_params['dataset']['batch_size'],
                                                                    num_pool=None,
                                                                    **trainer_params['dataset']['kwargs'])


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/cpu:0'):
            mdl_class = network.NETWORKS[trainer_params['model_name']]
            mdl = mdl_class(model_params)  # type: network.Network

            saver = tf.train.Saver()
            sw = tf.summary.FileWriter(model_dir)
            if os.path.isfile(os.path.join(model_dir, 'ckpt.meta')):
                saver.restore(sess, os.path.join(model_dir, 'ckpt'))
            else:
                sess.run(tf.global_variables_initializer())
                if len(sess.run(tf.report_uninitialized_variables())) > 0:
                    raise Exception('uninitialized variable exist !')

        with tf.device('/gpu:0'):
            # training
            train_placeholders, [step_op, global_step, total_loss] = mdl('train')
            # testing
            val_placeholders, [summary_op, global_step] = mdl('summarize')

        sess.run(tf.global_variables_initializer())

        sess.graph.finalize()

        for record in g_train:
            feed_dict = network.get_feed_dict(train_placeholders, record)
            _, current_step = sess.run([step_op, global_step], feed_dict)
            if current_step % trainer_params['summarize_step'] == 0:
                # perform summarize
                feed_dict = network.get_feed_dict(val_placeholders, g_val.next())
                summary, current_step = sess.run([summary_op, global_step], feed_dict)
                sw.add_summary(summary, current_step)
                saver.save(sess, os.path.join(model_dir, 'ckpt'))
                sw.flush()

def single_gpu_trainer_full(model_dir):
    with open(os.path.join(model_dir, 'model.json')) as f:
        model_params = json.load(f)

    with open(os.path.join(model_dir, 'trainer.json')) as f:
        trainer_params = json.load(f)

    g_txt_train = data.from_txt(trainer_params['dataset']['file_name'],
                                num_epoch=trainer_params['dataset']['num_epoch'],
                                num_skips=None,
                                is_train=True, do_shuffle=True)
    if trainer_params['dataset']['sampler'] is not None:
        g_txt_train = data.SAMPLERS[trainer_params['dataset']['sampler']](g_txt_train,
                                                                          **trainer_params['dataset']['sampler_config'])
    g_train = data.GENERATORS[trainer_params['dataset']['generator']](g_txt_train,
                                                                      batch_size=trainer_params['dataset']['batch_size'],
                                                                      num_pool=trainer_params['dataset']['num_pool'],
                                                                      **trainer_params['dataset']['kwargs'])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/cpu:0'):
            mdl_class = network.NETWORKS[trainer_params['model_name']]
            mdl = mdl_class(model_params)  # type: network.Network

            saver = tf.train.Saver()
            if os.path.isfile(os.path.join(model_dir, 'ckpt.meta')):
                saver.restore(sess, os.path.join(model_dir, 'ckpt'))
            else:
                sess.run(tf.global_variables_initializer())
                if len(sess.run(tf.report_uninitialized_variables())) > 0:
                    raise Exception('uninitialized variable exist !')

        with tf.device('/gpu:0'):
            # training
            train_placeholders, [step_op, global_step, total_loss] = mdl('train')

        sess.run(tf.global_variables_initializer())

        sess.graph.finalize()

        for record in g_train:
            feed_dict = network.get_feed_dict(train_placeholders, record)
            _, current_step, loss = sess.run([step_op, global_step, total_loss], feed_dict)
            if current_step % trainer_params['summarize_step'] == 0:
                saver.save(sess, os.path.join(model_dir, 'ckpt'))
                msg = 'Reached global step: {}, training loss: {}'.format(current_step, loss)
                with open('output.log', 'a') as f:
                    f.write(msg + '\n')

        with open('output.log', 'a') as f:
            f.write('Finished' + '\n')

