from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time
import glob
import os
from PIL import Image
from random import shuffle
from tqdm import tqdm
import cv2
import numpy as np


# Main slim library
import tensorflow.contrib.slim as slim
# from models.research.slim import nets
# from models.research.slim.nets import inception
import Project.inception_resnet_v2 as inception_resnet_v2
import Project.inception_preprocessing as inception_preprocessing
# from models.research.slim.datasets import dataset_utils

#================ DATASET INFORMATION ======================
dataset_dir = '/Project/train'
log_dir = 'log'
checkpoint_file = '/Project/inception_resnet_v2_2016_08_30.ckpt'
image_size = 299
num_classes = 10

labels_file = '/Project/labels.txt'
labels = open(labels_file, 'r')
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1]
    labels_to_name[int(label)] = string_name


file_pattern = 'dataset_tfrecord_%s_*.tfrecord'

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured scene image',
    'label': 'A label that is as such -- 0:airpot_inside, 1:bakery, 2:bedroom, 3:greenhouse, 4:gym, 5:kitchen, 6:operating_room, 7:poolinside, 8:restaurant, 9:toystore'
}

#================= TRAINING INFORMATION ==================
num_epochs = 20
batch_size = 64
initial_learning_rate = 0.00001
# learning_rate_decay_factor = 0.001
# num_epochs_before_decay = 5


# ============== DATASET LOADING ======================
def get_split_train(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='dataset_tfrecord'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later.
    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting
    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError(
            'The split_name %s is not recognized. Please input either train or validation as the split_name' % (
                split_name))

    # Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))
    print(file_pattern_path)
    # Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    print(file_pattern_for_counting)
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
                          file.startswith(file_pattern_for_counting)]
    print(tfrecords_to_count)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    print(num_samples)
    # Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def load_batch_train(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.
    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    '''
    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=batch_size,
        common_queue_min=24, shuffle=False)

    # Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])
    # Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    # As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, raw_images, labels

def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        dataset = get_split_train('train', dataset_dir, file_pattern=file_pattern)
        images, _, labels = load_batch_train(dataset, batch_size=batch_size)
        print(labels)
        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        print(num_steps_per_epoch)
        # decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(images, num_classes=dataset.num_classes,
                                                                         is_training=True)

        # Define the scopes that you want to exclude for restoration
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        #         lr = tf.train.exponential_decay(
        #             learning_rate = initial_learning_rate,
        #             global_step = global_step,
        #             decay_steps = decay_steps,
        #             decay_rate = learning_rate_decay_factor,
        #             staircase = True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', initial_learning_rate)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                # At the start of every epoch, show the vital information:
                if (step % num_batches_per_epoch) == 0:
                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    #                     learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    accuracy_value = sess.run([accuracy])
                    logging.info('Current Learning Rate: %s', initial_learning_rate)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)

                # Log the summaries every 10 step.
                if (step % 10) == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./inception_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    run()
