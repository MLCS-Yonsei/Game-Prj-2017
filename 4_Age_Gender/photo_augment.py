# coding=utf8
import os, errno
import cv2
import numpy as np

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to image")
ap.add_argument("-d", "--dir", type=str, default="",
	help="path to images")
ap.add_argument("-a", "--rotate_angle", nargs='+', type=int, default=[5, -5],
	help="path to images")
ap.add_argument("-g", "--gamma", nargs='+', type=float, default=[0.3, 1.6],
	help="path to images")
args = vars(ap.parse_args())

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

        # Change Gamma on Images
        for g in self.gamma:
            _decreased_gamma_img = self.adjust_gamma(_img, gamma=g)
            if _decreased_gamma_img is not False:
                save_path = os.path.join(augmented_dir,_id + '_g_' + str(g) + _ext)
                cv2.imwrite(save_path, _decreased_gamma_img)
                augmented_list.append(save_path)

        # # Save Horizontally flipped Image
        _horizontal_img = cv2.flip( _img, 1)
        save_path = os.path.join(augmented_dir,_id + '_f' + _ext)
        cv2.imwrite(save_path, _horizontal_img)
        augmented_list.append(save_path)

        # # Rotates both images
        _h, _w, _c = _img.shape

        _d = int(((_w*_w + _h*_h)**0.5)) 
        _rc = int(_w/2), int(_h/2)

        for a in self.rotate_angle:
            _angle = a    
            _scale = 1

            _r_image = cv2.getRotationMatrix2D(_rc, _angle, _scale)
            _rotate_img = cv2.warpAffine(_img, _r_image, (_w, _h))
            save_path = os.path.join(augmented_dir,_id + '_a_' + str(a) + _ext)
            cv2.imwrite(save_path ,_rotate_img)
            augmented_list.append(save_path)

            _r_image_f = cv2.getRotationMatrix2D(_rc, _angle, _scale)
            _rotate_img_f = cv2.warpAffine(_horizontal_img, _r_image, (_w, _h))
            save_path = os.path.join(augmented_dir,_id + '_f_a_' + str(a) + _ext)
            cv2.imwrite(save_path ,_rotate_img_f)
            augmented_list.append(save_path)
        
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

