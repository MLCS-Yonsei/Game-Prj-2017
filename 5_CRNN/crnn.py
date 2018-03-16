import numpy as np
import subprocess as sp
import os
import json
import conv_func
import tensorflow as tf
import cv2

# import matplotlib as plt

# dir = '/Users/jehyun/Dropbox/videos/'
dir = '/home/jhp/Dropbox/videos/'
filenames = os.listdir(dir)
# frame_number = 10; index = 0
# test_videos = np.empty([len(filenames)], dtype='object')
full_filename = os.path.join(dir,filenames[0])

cmnd = ['ffprobe', '-print_format', 'json', '-show_entries', 'stream=width,height', '-pretty', '-loglevel', 'quiet', full_filename]
p = sp.Popen(cmnd, stdout=sp.PIPE, stderr=sp.PIPE)

out, err =  p.communicate()

video_r = json.loads(out.decode('utf-8'))['streams'][0]
video_h = video_r['height']
video_w = video_r['width']

p.stdout.close()
p.terminate()

command = ['ffmpeg', '-loglevel', 'quiet','-i', full_filename, '-f','image2pipe', '-pix_fmt','rgb24', '-vcodec','rawvideo','-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

raw_image = pipe.stdout.read(video_h*video_w*3)
image = np.fromstring(raw_image, dtype='uint8')
image = image.reshape((video_h,video_w,3))
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = image.reshape((video_h*video_w))

pipe.stdout.flush()
pipe.terminate()

image = image.astype(np.float32)

c = conv_func.conv()

# inputdata = image
# inputdata = tf.cast(x=inputdata, dtype=tf.float32)
# feature_extraction(inputdata = inputdata)
# c.feature_extraction(inputdata = image)

batch_size = 128

x = tf.placeholder('float', [None, video_h*video_w])
y = tf.placeholder('float')

keep_prob = tf.placeholder(tf.float32)

output = c.convolutional_neural_network(image)
print(output.shape)

