import numpy as np
import subprocess as sp
import os
import json
import conv
import tensorflow as tf
import cv2

# import matplotlib.pyplot as plt

dir = '/Users/jehyun/Dropbox/videos/'
# dir = '/home/jhp/Dropbox/videos/'
filenames = os.listdir(dir)
frame_number = 10; index = 0; train_x=np.zeros((1,280000))

for filename in filenames:
    full_filename = os.path.join(dir,filename)

    cmnd = ['ffprobe', '-print_format', 'json', '-show_entries', 'stream=width,height', '-pretty', '-loglevel', 'quiet', full_filename]
    p = sp.Popen(cmnd, stdout=sp.PIPE, stderr=sp.PIPE)

    out, err =  p.communicate()

    video_r = json.loads(out.decode('utf-8'))['streams'][0]
    video_h = video_r['height']
    video_w = video_r['width']

    p.stdout.close()
    p.terminate()
    
    for i in range(frame_number):
        command = ['ffmpeg', '-loglevel', 'quiet','-i', full_filename, '-f','image2pipe', '-pix_fmt','rgb24', '-vcodec','rawvideo','-']
        pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

        raw_image = pipe.stdout.read(video_h*video_w*3)

        pipe.stdout.flush()
        pipe.terminate()

        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((video_h,video_w,3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.copyMakeBorder(image,0, 700-video_h,0, 400-video_w, cv2.BORDER_CONSTANT, value=[0,0,0])
        image = image.reshape((700*400))
        image = image.astype(np.float32)[np.newaxis, :]
        train_x = np.concatenate((train_x, image), axis =0)
        index +=1

train_x = train_x[1:,:]
np.savez_compressed('./data/train_x',a=train_x)
'''

a = np.ones(frame_number)[:,np.newaxis]; b = np.zeros(frame_number)[:,np.newaxis]
l1 = np.concatenate((a,b,b,b,b,b,b,b,b,b), axis = 1)
l2 = np.concatenate((b,a,b,b,b,b,b,b,b,b), axis = 1)
l3 = np.concatenate((b,b,a,b,b,b,b,b,b,b), axis = 1)
l4 = np.concatenate((b,b,b,a,b,b,b,b,b,b), axis = 1)
l5 = np.concatenate((b,b,b,b,a,b,b,b,b,b), axis = 1)
l6 = np.concatenate((b,b,b,b,b,a,b,b,b,b), axis = 1)
l7 = np.concatenate((b,b,b,b,b,b,a,b,b,b), axis = 1)
l8 = np.concatenate((b,b,b,b,b,b,b,a,b,b), axis = 1)
l9 = np.concatenate((b,b,b,b,b,b,b,b,a,b), axis = 1)
l10 = np.concatenate((b,b,b,b,b,b,b,b,b,a), axis = 1)
train_y = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10),axis = 0)
np.savez_compressed('train_y',a=train_y)

