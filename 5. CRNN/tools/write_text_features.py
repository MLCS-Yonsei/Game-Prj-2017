#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午7:47
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : write_text_features.py
# @IDE: PyCharm Community Edition
"""
Write text features into tensorflow records
"""
import os
import os.path as ops
import argparse
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from data_provider import data_provider
from local_utils import data_utils

import subprocess as sp
import json

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--save_dir', type=str, help='Where you store tfrecords')

    return parser.parse_args()


def write_features(dataset_dir, save_dir):
    """

    :param dataset_dir:
    :param save_dir:
    :return:
    """
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    print('Initialize the dataset provider ......')
    provider = data_provider.TextDataProvider(dataset_dir=dataset_dir, annotation_name='sample.txt',
                                              validation_set=True, validation_split=0.15, shuffle='every_epoch',
                                              normalization=None)
    print('Dataset provider intialize complete')

    feature_io = data_utils.TextFeatureIO()

    # write train tfrecords
    print('Start writing training tf records')

    # print(provider.train.images)
    train_videos = provider.train.videos
    for index, train_video in enumerate(train_videos):
        train_video = [cv2.resize(tmp, (100, 150)) for tmp in train_video]
        train_video = [bytes(list(np.reshape(tmp, [100 * 150 * 3]))) for tmp in train_video]

        train_videos[index] = train_video

    train_labels = provider.train.labels
    train_videonames = provider.train.videonames

    train_tfrecord_path = ops.join(save_dir, 'train_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=train_tfrecord_path, labels=train_labels, videos=train_videos,
                                     videonames=train_videonames)

    # write test tfrecords
    print('Start writing testing tf records')

    test_videos = provider.test.videos
    for index, test_video in enumerate(test_videos):

        test_video = [cv2.resize(tmp, (100, 150)) for tmp in test_video]
        test_video = [bytes(list(np.reshape(tmp, [100 * 150 * 3]))) for tmp in test_video]

        test_videos[index] = test_video

    test_labels = provider.test.labels
    test_videonames = provider.test.videonames

    test_tfrecord_path = ops.join(save_dir, 'test_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, videos=test_videos,
                                     videonames=test_videonames)

    # write val tfrecords
    print('Start writing validation tf records')

    val_videos = provider.validation.videos
    for index, val_video in enumerate(val_videos):
        
        val_video = [cv2.resize(tmp, (100, 150)) for tmp in val_video]
        val_video = [bytes(list(np.reshape(tmp, [100 * 150 * 3]))) for tmp in val_video]

        val_videos[index] = val_video

    val_labels = provider.validation.labels
    val_videonames = provider.validation.videonames

    val_tfrecord_path = ops.join(save_dir, 'validation_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=val_tfrecord_path, labels=val_labels, videos=val_videos,
                                     videonames=val_videonames)

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir)
