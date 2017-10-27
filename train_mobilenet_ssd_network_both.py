# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a SSD model using a given dataset."""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.97, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 8.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'smartphone_traffic', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 9, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', 'tf_records', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_ssd_traffic', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():

        # Create global_step.
        # with tf.device('/gpu:0'):
        global_step = slim.create_global_step()

        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        dataset_kitti = dataset_factory.get_dataset('kitti', FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=True)

        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers = FLAGS.num_readers,
                common_queue_capacity = 20 * FLAGS.batch_size,
                common_queue_min = 10 * FLAGS.batch_size,
                shuffle = True
            )
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape', 'object/label', 'object/bbox'])

        image, glabels, gbboxes = \
            image_preprocessing_fn(image, glabels, gbboxes, out_shape = ssd_shape, data_format = DATA_FORMAT)

        gclasses, glocalisations, gscores = \
            ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
        batch_shape = [1] + [len(ssd_anchors)] * 3

        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity= 5 * FLAGS.batch_size
        )

        b_image, b_gclasses, b_glocalisations, b_gscores = \
            tf_utils.reshape_list(r, batch_shape)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.image("imgs", tf.cast(b_image, tf.float32)))

        f_i = 0
        for gt_map in b_gscores:
            gt_features = tf.reduce_max(gt_map, axis=3)
            gt_features = tf.expand_dims(gt_features, -1)
            summaries.add(tf.summary.image("gt_map_%d" % f_i, tf.cast(gt_features, tf.float32)))
            f_i += 1
            # for festures in gt_list:
            #     summaries.add(tf.summary.image("gt_map_%d" % f_i, tf.cast(festures, tf.float32)))
            #     f_i += 1

        arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay, data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(b_image, is_training=True)

        f_i = 0
        for predict_map in predictions:
            predict_map = predict_map[:, :, :, :, 1:]
            predict_map = tf.reduce_max(predict_map, axis=4)
            predict_map = tf.reduce_max(predict_map, axis=3)
            predict_map = tf.expand_dims(predict_map, -1)
            summaries.add(tf.summary.image("predicte_map_%d" % f_i, tf.cast(predict_map, tf.float32)))
            f_i += 1

        ssd_net.losses(logits, localisations, b_gclasses, b_glocalisations, b_gscores,0,
                       match_threshold=FLAGS.match_threshold,
                       negative_ratio=FLAGS.negative_ratio,
                       alpha=FLAGS.loss_alpha,
                       label_smoothing=FLAGS.label_smoothing)

        with tf.name_scope('kitti' + '_data_provider'):
            provider_k = slim.dataset_data_provider.DatasetDataProvider(
                dataset_kitti,
                num_readers = FLAGS.num_readers,
                common_queue_capacity = 20 * FLAGS.batch_size,
                common_queue_min = 10 * FLAGS.batch_size,
                shuffle = True
            )
        [image_k, shape_k, glabels_k, gbboxes_k] = provider_k.get(['image', 'shape', 'object/label', 'object/bbox'])

        image_preprocessing_fn_k = preprocessing_factory.get_preprocessing('kitti', is_training=True)
        image_k, glabels_k, gbboxes_k = \
            image_preprocessing_fn_k(image_k, glabels_k, gbboxes_k, out_shape = ssd_shape, data_format = DATA_FORMAT)

        gclasses_k, glocalisations_k, gscores_k = \
            ssd_net.bboxes_encode(glabels_k, gbboxes_k, ssd_anchors)
        #batch_shape = [1] + [len(ssd_anchors)] * 3

        r_k = tf.train.batch(
            tf_utils.reshape_list([image_k, gclasses_k, glocalisations_k, gscores_k]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity= 5 * FLAGS.batch_size
        )

        b_image_k, b_gclasses_k, b_glocalisations_k, b_gscores_k = \
            tf_utils.reshape_list(r_k, batch_shape)

        summaries.add(tf.summary.image("k_imgs", tf.cast(b_image_k, tf.float32)))

        f_i = 0
        for gt_map in b_gscores_k:
            gt_features = tf.reduce_max(gt_map, axis=3)
            gt_features = tf.expand_dims(gt_features, -1)
            summaries.add(tf.summary.image("k_gt_map_%d" % f_i, tf.cast(gt_features, tf.float32)))
            f_i += 1
            # for festures in gt_list:
            #     summaries.add(tf.summary.image("gt_map_%d" % f_i, tf.cast(festures, tf.float32)))
            #     f_i += 1

        arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay, data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions_k, localisations_k, logits_k, end_points_k = \
                ssd_net.net(b_image_k, is_training=True, reuse=True)

        f_i = 0
        for predict_map in predictions_k:
            predict_map = predict_map[:, :, :, :, 1:]
            predict_map = tf.reduce_max(predict_map, axis=4)
            predict_map = tf.reduce_max(predict_map, axis=3)
            predict_map = tf.expand_dims(predict_map, -1)
            summaries.add(tf.summary.image("k_predicte_map_%d" % f_i, tf.cast(predict_map, tf.float32)))
            f_i += 1

        ssd_net.losses(logits_k, localisations_k, b_gclasses_k, b_glocalisations_k, b_gscores_k, 0,
                       match_threshold=FLAGS.match_threshold,
                       negative_ratio=FLAGS.negative_ratio,
                       alpha=FLAGS.loss_alpha,
                       label_smoothing=FLAGS.label_smoothing)



        #total_loss = slim.losses.get_total_loss()
        total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.scalar('loss', total_loss))

        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        # for variable in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))
        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))


        learning_rate = tf_utils.configure_learning_rate(FLAGS, dataset.num_samples, global_step)
        optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.adam_beta1,
        #                                    beta2=FLAGS.adam_beta2, epsilon=FLAGS.opt_epsilon)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=False
        )

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options = gpu_options)

        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            init_fn=tf_utils.get_init_fn(FLAGS),
            #summary_op = summary_op,
            number_of_steps = FLAGS.max_number_of_steps,
            log_every_n_steps = FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver = saver,
            save_interval_secs = FLAGS.save_interval_secs,
            session_config = config,
            sync_optimizer=None
        )






if __name__ == '__main__':
    tf.app.run()
