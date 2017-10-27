import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.kitti_common import KITTI_LABELS

DIRECTORY_ANNOTATIONS = 'label_xml/'
DIRECTORY_IMAGES = 'image_2/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def _process_image(directory, name):

    filename = directory + DIRECTORY_IMAGES + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    bboxes = []
    alphas = []
    labels = []
    labels_text = []
    truncated = []
    occluded = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(KITTI_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        alphas.append(float(obj.find('alpha').text))
        truncated.append(float(obj.find('truncated').text))
        occluded.append(int(obj.find('occluded').text))
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1],
                      ))

    return image_data, shape, bboxes, labels, labels_text, alphas, truncated, occluded

def _convert_to_example(image_data, labels, labels_text, bboxes, shape, alphas, truncated, occluded):

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape' : int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/alpha': float_feature(alphas),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/truncated': float_feature(truncated),
        'image/object/bbox/occluded': int64_feature(occluded),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))

    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, alphas, truncated, occluded = _process_image(dataset_dir, name)

    example = _convert_to_example(image_data, labels, labels_text, bboxes, shape, alphas, truncated, occluded)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, name='kitti_train', shuffling=False):

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r >> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print ('\nFinished converting the kitti dataset!')