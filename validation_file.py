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
# State your log directory where you can retrieve your model
log_dir = 'log'  # '/content/drive/My Drive/models_sarah'
dataset_dir = '/Project/train'
batch_size = 683

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


def run2():
    # Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split_train('validation', dataset_dir)
        images, raw_images, labels = load_batch_train(dataset, batch_size=batch_size, is_training=False)
        #         images, raw_images, labels, names = load_batch(dataset, batch_size = batch_size, is_training = False)
        #         print(len(names))
        # Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        # Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(images, num_classes=dataset.num_classes,
                                                                         is_training=False)

        probabilities = tf.nn.softmax(logits)

        checkpoint_path = tf.train.latest_checkpoint(log_dir)
        print(checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            slim.get_variables_to_restore())
        count = 0

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):

                sess.run(tf.initialize_local_variables())
                init_fn(sess)
                np_probabilities, np_images_raw, np_labels = sess.run([probabilities, raw_images, labels])

                for i in range(batch_size):
#                     image = np_images_raw[i, :, :, :]
                    true_label = np_labels[i]
                    predicted_label = np.argmax(np_probabilities[i, :])
                    predicted_name = dataset.labels_to_name[predicted_label]
                    true_name = dataset.labels_to_name[true_label]

                    if (true_name == predicted_name):
                        count = count + 1

                print(count)
                print((count / batch_size))
                print((count / batch_size) * 100)

if __name__ == '__main__':
    run2()
