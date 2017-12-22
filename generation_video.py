import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('./')

from nets import ssd_vgg, ssd_common, np_methods
from nets import mobilenet_pretrained_owndata
from preprocessing import ssd_owndata_preprocessing

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (440, 440)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_owndata_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_owndata_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
# reuse = True if 'ssd_net' in locals() else None
# ssd_net = ssd_vgg.SSDNet()
# with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
#     predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
reuse = True if 'MobilenetV1' in locals() else None
ssd_params = mobilenet_pretrained_owndata.Mobilenet_SSD_Traffic.default_params._replace(
            num_classes=11)
ssd_net = mobilenet_pretrained_owndata.Mobilenet_SSD_Traffic(ssd_params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.image("Image", image_4d))
    # f_i = 0
    # for predict_map in predictions:
        # predict_map = predict_map[:, :, :, :, 1:]
        # predict_map = tf.reduce_max(predict_map, axis=4)
        # if f_i < 3:
            # predict_list = tf.split(predict_map, 6, axis=3)
            # anchor_index = 1
            # for anchor in predict_list:
                # summaries.add(tf.summary.image("predicte_map_%d_anchor%d" % (f_i,anchor_index), tf.cast(anchor, tf.float32)))
                # anchor_index += 1
        # else:
            # predict_map = tf.reduce_max(predict_map, axis=3)
            # predict_map = tf.expand_dims(predict_map, -1)
            # summaries.add(tf.summary.image("predicte_map_%d" % f_i, tf.cast(predict_map, tf.float32)))
        # f_i += 1
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# #ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
# isess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(isess, ckpt_filename)
#ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs_scre/checkpoint'))
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))
# if that checkpoint exists, restore from checkpoint
saver = tf.train.Saver()
summer_writer = tf.summary.FileWriter("./logs_test/", isess.graph)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(isess, ckpt.model_checkpoint_path)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.35, nms_threshold=0.35, net_shape=(1024, 1024)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img, summary_op_str = isess.run([image_4d, predictions, localisations, bbox_img, summary_op],
                                                              feed_dict={img_input: img})


    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=11, decode=True)

    # print ("shape of rclasses:", rclasses.shape)
    # print ("shape of rscores:", rscores.shape)
    # print ("shape of rbboxes:", rbboxes.shape)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    # print ('====================')
    # print ("shape of rclasses:", rclasses.shape)
    # print ("shape of rscores:", rscores.shape)
    # print ("shape of rbboxes:", rbboxes.shape)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # print ("~~~~~~~~~~~~~~~~~~~~")
    # print ("shape of rclasses:", rclasses.shape)
    # print ("shape of rscores:", rscores.shape)
    # print ("shape of rbboxes:", rbboxes.shape)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    summer_writer.add_summary(summary_op_str, 1)
    # print rclasses, rscores, rbboxes
    return rclasses, rscores, rbboxes
color_list = [(0, 0, 255), (255, 0, 0), (255, 0, 0), (255, 255, 0),
              (255, 255,0), (255, 255, 0), (255, 255, 0), (0, 255, 0),
              (0, 255,0), (0, 255, 0), (0, 255, 255), (0, 255, 0)]

def draw_results(img, rclasses, rscores, rbboxes, index):

    # height = 1024
    # width = 1024
    height, width, channels = img.shape[:3]
    for i in range(len(rclasses)):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_list[rclasses[i]], thickness=2)
        cv2.putText(img, str(rclasses[i]), (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[rclasses[i]])
        if ((2 in rclasses) or (1 in rclasses) or (3 in rclasses) or
                (5 in rclasses) or (6 in rclasses) or (4 in rclasses)):
            cv2.putText(img, "Intersection", (180, 100), cv2.FONT_ITALIC, 2,
                        (255, 255, 0))
        else:
            cv2.putText(img, "Road", (180, 100), cv2.FONT_ITALIC, 2,
                        (255, 255, 0))
    #cv2.imshow("predict", img)
    # cv2.imwrite('./test_result/test_%d.jpg' % index, img)
    return img

# path = '/home/gpu_server2/DataSet/dayTrain/dayTest/daySequence1/frames/'
# path = './test_img/'
# image_names = sorted(os.listdir(path))
# print image_names
# index = 1
# for image_name in image_names:
    # img = cv2.imread(path + image_name)
    # destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rclasses, rscores, rbboxes =  process_image(destRGB)
    # draw_results(img, rclasses, rscores, rbboxes, index)
    # index += 1

capture = cv2.VideoCapture('./Audio201605231133.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = capture.read()
h, w, _ = frame.shape
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (w,h))
count = 0
for i in range(8000):
    ret, frame = capture.read()
    count = count + 1
    if count > 2400: 
        destRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rclasses, rscores, rbboxes =  process_image(destRGB)
        img = draw_results(frame, rclasses, rscores, rbboxes, count)
        out.write(img)
        print count
        # pic_name = str(i) +'.jpg'
        # cv2.imwrite(img_folder + pic_name , frame)
        # print (pic_name)

capture.release()
out.release()
