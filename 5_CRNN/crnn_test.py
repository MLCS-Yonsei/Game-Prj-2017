# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
# from datetime import datetime
# import hashlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import random
# import re
# import struct
# import sys
# import tarfile

import numpy as np
# from six.moves import urllib
import tensorflow as tf

# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
# from tensorflow.python.util import compat

model_dir = '/home/jehyunpark/Downloads/crnn/results/imagenet'
image_path = '/home/jehyunpark/Downloads/crnn/images/handwaving/'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
i = 0

filenames = sorted(os.listdir(image_path), key = lambda a:a[6:11])[:10]

# frame1 = os.path.join(image_path,'person01_01.jpg')
# frame2 = os.path.join(image_path,'person01_02.jpg')
# frame3 = os.path.join(image_path,'person01_03.jpg')
# frame4 = os.path.join(image_path,'person01_04.jpg')
# frame5 = os.path.join(image_path,'person01_05.jpg')
# frame6 = os.path.join(image_path,'person01_06.jpg')
# frame7 = os.path.join(image_path,'person01_07.jpg')
# frame8 = os.path.join(image_path,'person01_08.jpg')
# frame9 = os.path.join(image_path,'person01_09.jpg')
# frame10 = os.path.join(image_path,'person01_10.jpg')

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(
        model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.
  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.
  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

with tf.Session(graph=graph) as sess:
  for filename in filenames:
    full_filename = os.path.join(image_path,filename)
    if i == 0:
      jpeg_data = gfile.FastGFile(full_filename, 'rb').read()
      frames = run_bottleneck_on_image(sess, jpeg_data, jpeg_data_tensor, bottleneck_tensor)[np.newaxis,:]
      i +=1
    elif len(frames) < 10:
      jpeg_data = gfile.FastGFile(full_filename, 'rb').read()
      frames = np.concatenate((frames, run_bottleneck_on_image(sess, jpeg_data, jpeg_data_tensor, bottleneck_tensor)[np.newaxis,:]), axis = 0)
      # i +=1
  
  
  print(frames.shape)