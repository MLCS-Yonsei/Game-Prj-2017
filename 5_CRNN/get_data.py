import numpy as np
import subprocess as sp
import os
import json
import tensorflow as tf
import cv2


# dir = '/Users/jehyun/Dropbox/videos/'
# dir = '/home/jhp/Dropbox/videos/'
dir = '/home/hwanmooy/code/google-AVA-Dataset-downloader-master/data/train'
save_dir = '/home/jehyunpark/data/'

filenames = os.listdir(dir)
#filenames = filenames[:100]
frame_number = 1; index = 0; train_x=np.zeros((1,1262*720)); train_y = np.zeros((1,10))

l1 = np.array([1,0,0,0,0,0,0,0,0,0])[np.newaxis, :]
l2 = np.array([0,1,0,0,0,0,0,0,0,0])[np.newaxis, :]
l3 = np.array([0,0,1,0,0,0,0,0,0,0])[np.newaxis, :]
l4 = np.array([0,0,0,1,0,0,0,0,0,0])[np.newaxis, :]
l5 = np.array([0,0,0,0,1,0,0,0,0,0])[np.newaxis, :]
l6 = np.array([0,0,0,0,0,1,0,0,0,0])[np.newaxis, :]
l7 = np.array([0,0,0,0,0,0,1,0,0,0])[np.newaxis, :]
l8 = np.array([0,0,0,0,0,0,0,1,0,0])[np.newaxis, :]
l9 = np.array([0,0,0,0,0,0,0,0,1,0])[np.newaxis, :]
l10 = np.array([0,0,0,0,0,0,0,0,0,1])[np.newaxis, :]
label = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10),axis = 0)

for filename in filenames:
    a = filename.split('_')
    b = float(a[5])
    if b%1 !=0:
        b = int(a[6])
    else:
        b = int(a[5])
    if 0 < b <11:
        full_filename = os.path.join(dir,filename)

        cmnd = ['ffprobe', '-print_format', 'json', '-show_entries', 'stream=width,height', '-pretty', '-loglevel', 'quiet', full_filename]
        p = sp.Popen(cmnd, stdout=sp.PIPE, stderr=sp.PIPE)

        out, err =  p.communicate()

        video_r = json.loads(out.decode('utf-8'))['streams'][0]
        video_h = video_r['height']
        video_w = video_r['width']

        p.stdout.close()
        p.terminate()
        # print(video_h, video_w)
        command = ['ffmpeg', '-loglevel', 'quiet','-i', full_filename, '-f','image2pipe', '-pix_fmt','rgb24', '-vcodec','rawvideo','-']
        pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
        
        for i in range(frame_number):
            raw_image = pipe.stdout.read(video_h*video_w*3)
            
            image = np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((video_h,video_w,3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.copyMakeBorder(image,0, 720-video_h,0, 1262-video_w, cv2.BORDER_CONSTANT, value=[0,0,0])
            image = image.reshape((1262*720))
            image = image.astype(np.float32)[np.newaxis, :]
            train_x = np.concatenate((train_x, image), axis =0)

        pipe.stdout.flush()
        pipe.terminate()


        train_y = np.concatenate((train_y, label[b-1,:][np.newaxis, :]),axis = 0)
        index +=1
        if index%100 ==0:
            print(index)

train_x = train_x[1:,:]
train_y = train_y[1:,:]
np.savez_compressed(save_dir + 'train_x',a=train_x)
np.savez_compressed(save_dir + 'train_y',a=train_y)

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
np.savez_compressed('./data/train_y',a=train_y)

'''