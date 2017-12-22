import numpy as np
import cv2
import time
import sys
from InverseMapping import IPM
import glob
import matplotlib.pyplot as plt

def transform_topview(image_folder):
    img_names = glob.glob('./{}/*.png'.format(image_folder))
    ipm = None
    for filename in img_names:
        
        if (filename.count('17318')):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, ch = img.shape
            d_height = height * 3 
            d_width = width / 2
            M = cv2.getRotationMatrix2D((height/2, width/2), 2, 1)
            img = cv2.warpAffine(img, M, (width, height))
            plt.imshow(img)
            points = plt.ginput(4, timeout=160)
            ori_points = np.float32([points[0], points[1], points[2], points[3]])
            print ori_points
            plt.close()
            
            
            # ori_points = np.float32([[100, height],[width - 100, height],[width/2+35, 360], [width/2-20, 360]])
            # print ori_points
            dst_points = np.float32([[0,d_height],[d_width,d_height],[d_width, 0], [0, 0]])
            np.savez('transform_para.npz', ori_points, dst_points)
            ipm = IPM((width, height), (d_width, d_height), ori_points, dst_points)
            output = ipm.transform(img)
            plt.imshow(output)
            plt.show()
            break

    for filename in img_names:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        M = cv2.getRotationMatrix2D((height/2, width/2), 2, 1)
        img = cv2.warpAffine(img, M, (width, height))
        output = ipm.transform(img)
        plt.imshow(output)
        plt.show()

transform_topview('test_img')

