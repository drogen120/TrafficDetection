import cv2
import numpy as np

class IPM():

    def __init__(self, ori_imgsize, dst_imgsize, origPoints, dstPoints):
        self.ori_imgsize = ori_imgsize
        self.dst_imgsize = dst_imgsize
        self.ori_points = origPoints
        self.dst_points = dstPoints
        self.mH = cv2.getPerspectiveTransform(self.ori_points, self.dst_points)
        self.minvH = cv2.getPerspectiveTransform(self.dst_points, self.ori_points)

    def transform(self, inputimg):
        dst = cv2.warpPerspective(inputimg, self.mH, self.dst_imgsize)
        return dst

    def inv_transform(self, inputimg):
        dst = cv2.warpPerspective(inputimg, self.minvH, self.ori_imgsize)
        return dst


# cap = cv2.VideoCapture('Audio201606061616.mp4')
# ret, frame = cap.read()
# height, width, ch = frame.shape
# ori_points = np.float32([[10,height],[width - 10,height],[width/2+70, 325], [width/2-30, 325]])
# dst_points = np.float32([[0,height],[width,height],[width, 0], [0, 0]])
# #ori_points = np.float32([[10,height],[width - 10,height],[width/2+70, 325]])
# #dst_points = np.float32([[0,height],[width,height],[width, 0]])
# print (tuple(ori_points[0]))
# ipm = IPM((width, height), (width, height), ori_points, dst_points)
# while(1):
#     ret, frame = cap.read()
#     if(ret):
#         output = ipm.transform(frame)
#         cv2.line(frame, tuple(ori_points[0]), tuple(ori_points[1]), (255,0,0), 3)
#         cv2.line(frame, tuple(ori_points[1]), tuple(ori_points[2]), (255,0,0), 3)
#         cv2.line(frame, tuple(ori_points[2]), tuple(ori_points[3]), (255,0,0), 3)
#         cv2.line(frame, tuple(ori_points[3]), tuple(ori_points[0]), (255,0,0), 3)
#         cv2.imshow('origin',frame)
#         #gre = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#         #th3 = cv2.adaptiveThreshold(gre,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#         cv2.imshow('transeformed',output)
#         k = cv2.waitKey(30) & 0xff
#     else:
#         break