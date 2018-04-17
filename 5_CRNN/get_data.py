import numpy as np
import subprocess as sp
import os
import json
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile



save_dir = '/home/jehyunpark/data/'
model_dir = '/home/jehyunpark/Downloads/crnn/results/imagenet'
image_path = '/home/jehyunpark/Downloads/crnn/images/boxing/'

filenames = os.listdir(dir)
filenames = filenames[51014:]
frame_number = 10; index = 51014; index2 = 0; n_classes = 10
train_x=np.zeros((1,1262*720)); train_y = np.zeros((1,10))




def create_inception_graph():
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
    RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def one_hot(label, n_classes):
    a = np.zeros(n_classes)[np.newaxis, :]
    a[0,label] = 1
    return a

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

        train_y = np.concatenate((train_y, one_hot(b-1,n_classes)),axis = 0)
        index2 +=1
        if index2%100 == 0:
            print('-------------')
            np.savez_compressed(save_dir + 'train_x'+str(index),a=train_x)
            np.savez_compressed(save_dir + 'train_y'+str(index),a=train_y)
    index +=1
    if index%100 == 0:
        print(index)

train_x = train_x[1:,:]
train_y = train_y[1:,:]
np.savez_compressed(save_dir + 'train_x'+str(index),a=train_x)
np.savez_compressed(save_dir + 'train_y'+str(index),a=train_y)

