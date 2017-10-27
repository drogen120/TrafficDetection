import os
import sys
import random
import csv

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.LISA_common import LISA_LABELS

DIRECTORY_ANNOTATIONS = '%s/frameAnnotationsBOX.xml'
DIRECTORY_IMAGES = '%s/frames/'
dataset_folder = '/home/gpu_server2/DataSet/dayTrain/'
img_folder_list = ["dayClip1","dayClip2","dayClip3","dayClip4","dayClip5","dayClip6","dayClip7","dayClip8","dayClip9","dayClip10","dayClip11","dayClip12","dayClip13"]
# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

annotion_list = []
# def _read_annotation_file(filename):
#     with open('filename', 'rb') as csvfile:
#         annotion_reader = csv.reader(csvfile, delimiter = ';')
#         for row in annotion_reader:
#             annotion_list.append(row)

def _process_image(directory, folder_name ,name):

    filename = directory + DIRECTORY_IMAGES % folder_name + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    filename = directory + DIRECTORY_ANNOTATIONS % folder_name
    tree = ET.parse(filename)
    root = tree.getroot()

    shape = [960, 1280, 3]
    bboxes = []
    labels = []
    labels_text = []

    for image_root in root.findall(name + '.png'):
        #print "debug1"
        for obj in image_root.findall('object'):
            #print "debug2"
            label = obj.find('Annotation_tag').text
            labels.append(int(LISA_LABELS[label][0]))
            labels_text.append(label.encode('ascii'))
            bboxes.append((float(obj.find('Upper_left_corner_Y').text) / shape[0],
                           float(obj.find('Upper_left_corner_X').text) / shape[1],
                           float(obj.find('Lower_right_corner_Y').text) / shape[0],
                           float(obj.find('Lower_right_corner_X').text) / shape[1],
                          ))
    return image_data, shape, bboxes, labels, labels_text

def _convert_to_example(image_data, shape, bboxes, labels, labels_text):
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
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))
    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer, folder):
    image_data, shape, bboxes, labels, labels_text = _process_image(dataset_dir, folder, name)
    example = _convert_to_example(image_data, shape, bboxes, labels, labels_text)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, name='lisa_train', shuffling=False):

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    fidx = 0
    for folder in img_folder_list:
        i = 0
        path = os.path.join(dataset_dir, DIRECTORY_IMAGES % folder)
        filenames = sorted(os.listdir(path))
        if shuffling:
            random.seed(RANDOM_SEED)
            random.shuffle(filenames)
        while i < len(filenames):
            tf_filename = _get_output_filename(output_dir, name, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(filenames) and j < SAMPLES_PER_FILES:
                    sys.stdout.write('\r >> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    filename = filenames[i]
                    img_name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer, folder)
                    i += 1
                    j += 1
                fidx += 1

    print ('\nFinished converting the kitti dataset!')
