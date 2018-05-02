# coding=utf8
import os, errno
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

import face_alignment
from skimage import io

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="./test.jpg",
	help="path to image")
ap.add_argument("-d", "--dir", type=str, default="",
	help="path to images")
ap.add_argument("-a", "--rotate_angle", nargs='+', type=int, default=[5, -5],
	help="path to images")
ap.add_argument("-g", "--gamma", nargs='+', type=float, default=[0.3, 1.6],
	help="path to images")
ap.add_argument("-p", "--shape-predictor", default="./bin/shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

class photoAugmentor:
    def __init__(self, rotate_angle, gamma):
        self.isList = False
        self.rotate_angle = rotate_angle # [5, -5]
        self.gamma = gamma

    def generate(self, input):
        if isinstance(input, list):
            self.isList = True

        path = os.path.normpath(input)
        original_dir, img = os.path.split(path)
        augmented_dir = os.path.join(original_dir,'augmented')

        try:
            os.makedirs(augmented_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        _id = os.path.splitext(img)[0]
        _ext = os.path.splitext(img)[1]

        augmented_list = []

        # Copy Original Image
        _img = cv2.imread(os.path.join(original_dir,img))
        save_path = os.path.join(augmented_dir,_id + _ext)
        cv2.imwrite(save_path, _img)
        augmented_list.append(save_path)

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)

        input = io.imread(os.path.join(original_dir,img))
        preds = fa.get_landmarks(input)[0]

        print(preds)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(preds[:,0],preds[:,1],preds[:,2], 'bo')
        plt.show()
        return augmented_list

    def adjust_gamma(self, image, gamma=1.0):
        if gamma > 0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")

            return cv2.LUT(image, table)

        return false

if __name__ == '__main__':
    pa = photoAugmentor(args["rotate_angle"], args["gamma"])
    path = args["image"]
    r = pa.generate(path)

    print(r)

