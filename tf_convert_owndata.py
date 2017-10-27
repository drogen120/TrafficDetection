import tensorflow as tf

from datasets import kitti_to_tfrecords
from datasets import smartphone_traffic_to_tfrecords

def main(_):

    # kitti_to_tfrecords.run('./kitti_data/', './tf_records', 'kitti_train', shuffling = True)

    smartphone_traffic_to_tfrecords.run('./autolabel/', './tf_records',
                                        'owndata_train', shuffling = True)
if __name__ == '__main__':
    tf.app.run()
