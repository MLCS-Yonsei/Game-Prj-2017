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
image_path = '/home/jehyunpark/Downloads/crnn/images/handwaving/person01_01.jpg'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

# test_sample = np.load('./results/bottleneck/handwaving/person01_01.jpg.txt.npz')['a']

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
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # image_data = sess.run(image,{input_jpeg_tensor: jpeg_data})
    # image_data = sess.run(image,{jpeg_data_tensor: jpeg_data})
    bottleneck_values = run_bottleneck_on_image(sess, jpeg_data, jpeg_data_tensor, bottleneck_tensor)
    print(bottleneck_values.shape)