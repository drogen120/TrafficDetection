import os
import cv2
from os import walk
import numpy as np
import tqdm
import xml.etree.ElementTree as ET
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['KITTI']

KITTI_LABELS = {
    'DontCare': (0, 'Background'),
    'Car': (1, 'Vehicle'),
    'Van': (2, 'Vehicle'),
    'Truck': (3, 'Vehicle'),
    'Pedestrian': (4, 'Person'),
    'Person_sitting': (5, 'Person'),
    'Cyclist': (6, 'Person'),
    'Tram': (7, 'Vehicle'),
    'Misc': (8, 'Vehicle'),
}

def get_training_bbox(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    bboxes = []
    alphas = []
    labels = []
    # labels_text = []
    truncated = []
    occluded = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(KITTI_LABELS[label][0]))
        # labels_text.append(label.encode('ascii'))

        alphas.append(float(obj.find('alpha').text))
        truncated.append(float(obj.find('truncated').text))
        occluded.append(int(obj.find('occluded').text))
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1],
                      ))

    return np.asarray(labels), np.asarray(alphas), \
            np.asarray(truncated), np.asarray(occluded)

class KITTI(RNGDataFlow):
    
    def __init__(self, dir , name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'

        self.image_list = []
        self.image_path = self.full_dir + '/image/'
        for (dirpath, dirnames, filenames) in walk(self.image_path):
            self.image_list.extend(filenames)
            break

        self.label_list = []
        self.label_path = self.full_dir + '/label/'
        for (dirpath, dirnames, filenames) in walk(self.label_path):
            self.label_list.extend(filenames)
            break

        self.image_list.sort()
        self.label_list.sort()

    def size(self):
        return len(self.image_list)

    def get_data(self):
        idxs = np.arange(len(self.image_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_name = os.path.join(self.image_path, self.image_list[k])
            img = cv2.imread(img_name)
            label_name = os.path.join(self.label_path, self.label_list[k])
            # annotions = get_training_bbox(label_name)
            yield img, get_training_bbox(label_name)

if __name__ == '__main__':
    ds = KITTI('./kitti_train/', 'train', shuffle = True)
    print ds.image_path, ds.label_path
    ds.reset_state()

    for k in ds.get_data():
        # from IPython import embed
        # embed()
        print k
        break
