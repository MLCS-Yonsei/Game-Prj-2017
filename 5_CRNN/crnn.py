import numpy as np
import subprocess as sp
import os
import json
import conv_func
import tensorflow as tf

# import matplotlib as plt

dir = '/Users/jehyun/Dropbox/videos/'
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

pipe.stdout.flush()
pipe.terminate()

c = conv_func.conv()

def feature_extraction(inputdata):
    conv1 = c.conv2d(inputdata=inputdata, out_channel=64, kernel_size=3, stride=1, use_bias=False, name='conv1') 
    relu1 = c.relu(inputdata=conv1)
    max_pool1 = c.maxpooling(inputdata=relu1, kernel_size=2, stride=2)
    conv2 = c.conv2d(inputdata=max_pool1, out_channel=128, kernel_size=3, stride=1, use_bias=False, name='conv2') 
    relu2 = c.relu(inputdata=conv2)
    max_pool2 = c.maxpooling(inputdata=relu2, kernel_size=2, stride=2)
    conv3 = c.conv2d(inputdata=max_pool2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3')  # batch*8*25*256 #66*27*256
    relu3 = c.relu(conv3) # batch*8*25*256
    conv4 = c.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*25*256 #66*27*256
    relu4 = c.relu(conv4)  # batch*8*25*256
    max_pool4 = c.maxpooling(inputdata=relu4, kernel_size=[4, 1], stride=[4, 1], padding='VALID')  # batch*4*25*256 #16*27*256
    conv5 = c.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*25*512 #16*27*512 
    relu5 = c.relu(conv5)

    return relu5
inputdata = image
inputdata = tf.cast(x=inputdata, dtype=tf.float32)
feature_extraction(inputdata = inputdata)




