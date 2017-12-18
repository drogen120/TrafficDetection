import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image
import cv2
from nets import mobilenet_pretrained_owndata
from nets import nets_factory
from nets import tf_methods
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from preprocessing import ssd_owndata_preprocessing


def draw_results(img, rclasses, rscores, rbboxes):
    img = np.array(img)
    height, width, channels = img.shape[:3]
    for i in range(len(rclasses)):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
        cv2.putText(img, str(rclasses[i]), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
        # cv2.putText(img, str(rscores[i]), (xmin, ymin),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
    cv2.imwrite('./pb_convert_img/test.jpg', img)

data_format = 'NHWC'
net_shape = (440, 440)
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))
select_threshold = 0.20
nms_threshold = 0.45

with tf.Graph().as_default() as graph:
    input_tensor = tf.placeholder(tf.uint8, shape=(None, 440, 440, 3),
                                  name='image_tensor')
    # input_for_preprocess = tf.squeeze(input_tensor)
    # only chooses the first frame
    input_for_preprocess = input_tensor[0]
    input_for_preprocess = tf.cast(input_for_preprocess, tf.float32)
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_owndata_preprocessing.preprocess_for_eval(
        input_for_preprocess, None, None, net_shape, data_format, resize=ssd_owndata_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)
    print (image_4d.get_shape())

    with tf.Session() as sess:
        ssd_params = mobilenet_pretrained_owndata.Mobilenet_SSD_Traffic.default_params._replace(
                num_classes=11)
        ssd_net = mobilenet_pretrained_owndata.Mobilenet_SSD_Traffic(ssd_params)
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)
            # print ('len of predictions :', len(predictions))
            # print ('predictions shape:', predictions[0].shape)
            ssd_anchors = ssd_net.anchors(net_shape)

            classes, scores, _bboxes = tf_methods.tf_ssd_bboxes_select(
                    predictions, localisations, ssd_anchors,
                    select_threshold=select_threshold, img_shape=net_shape,
                    num_classes=11, decode=True)

            # print ('shape pf classes:',classes.shape)
            # print ('shape of scores:',scores.shape)
            # print ('shape of _bboxes:',_bboxes.shape)
            # print ('**************')
            # raise

            bboxes = tf_methods.tf_bboxes_clip(bbox_img, _bboxes)
            # classes = tf.squeeze(_classes)
            # scores = tf.squeeze(_scores)
            # print ('shape pf classes:',classes.shape)
            # print ('shape of scores:',scores.shape)
            # print ('shape of bboxes:',bboxes.shape)
            # print ('**************')
            # I just comment this !!!!!!!!!!!!!!!
            # some bugs in sort methods need to be solved !!!!!!
            classes = tf.expand_dims(classes, axis=0)
            scores = tf.expand_dims(scores, axis=0)
            bboxes = tf.expand_dims(bboxes , axis=0)

            classes, scores, bboxes = tf_methods.tf_bboxes_sort(classes,
                                                                scores, bboxes,
                                                                top_k=100)
            # print ('shape pf classes:',classes.shape)
            # print ('shape of scores:',scores.shape)
            # print ('shape of bboxes:',bboxes.shape)
            # print ('**************')
            # classes = tf.squeeze(classes)
            # scores = tf.squeeze(scores)
            # bboxes = tf.squeeze(bboxes)
            classes, scores, bboxes, end_index = tf_methods.tf_bboxes_nms(classes, scores, bboxes, nms_threshold=nms_threshold)
            # print ('shape pf classes:',classes.shape)
            # print ('shape of scores:',scores.shape)
            # print ('shape of bboxes:',bboxes.shape)
            # print ('**************')
            classes = tf.squeeze(classes)
            scores = tf.squeeze(scores)
            bboxes = tf.squeeze(bboxes)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            bboxes = tf_methods.tf_bboxes_resize(bbox_img, bboxes)

            scores_ = tf.identity(scores,'detection_scores')
            bboxes_ = tf.identity(bboxes,'detection_boxes')
            classes_ = tf.identity(classes,'detection_classes')

        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


        # try feed forward graph
        file_path = './test_img/201702071403_00024341.png'
        image = Image.open(file_path)
        img = image.resize((440,440), Image.ANTIALIAS)
        input_image = np.expand_dims(np.array(img),0)
        # run threshold selected nodes
        # rclasses, rscores, rbboxes,rpredictions, rlocalisations = sess.run(
            # [_classes,_scores,_bboxes,predictions, localisations], feed_dict={
            # input_tensor: input
        # })
        # draw_results(img,rclasses,rscores,rbboxes)
        # print ('shape of rpredictions:',rpredictions[0].shape)
        # print ('shape of rlocalisations:',rlocalisations[0].shape)
        # print (rclasses)
        # print ('shape of rclasses:',rclasses.shape)
        # print (rscores)
        # print ('shape of rscores:',rscores.shape)
        # print (rbboxes)
        # print ('shape of rbboxes:',rbboxes.shape)
        # run after nms nodes
        rclasses_,rscores_, rbboxes_, index_str = sess.run(
            [classes_,scores_,bboxes_, end_index], feed_dict={
            input_tensor: input_image
        })
        print('rscores_:',rscores_)
        print('rbboxes:',rbboxes_)
        print('rclasses_:',rclasses_)
        print('index_end:',index_str)
        draw_results(image, rclasses_, rscores_, rbboxes_)
        # raise

        output_node_names = 'detection_classes,detection_boxes,detection_scores'
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))

        # with open('./output_graph_nodes_prepro.txt', 'w') as f:
        #     for node in output_graph_def.node:
        #         f.write(node.name + '\n')
        if not os.path.exists('./tf_files'):
            os.mkdir('./tf_files')
        output_graph = './tf_files/traffic_detection.pb'
        with gfile.FastGFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
