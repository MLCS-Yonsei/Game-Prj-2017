import numpy as np
import subprocess as sp
import os
import json
import conv
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt

# dir = '/Users/jehyun/Dropbox/videos/'
dir = '/home/jhp/Dropbox/videos/'
filenames = os.listdir(dir)
frame_number = 10; index = 0
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

image = cv2.copyMakeBorder(image,0, 700-video_h,0, 400-video_w, cv2.BORDER_CONSTANT, value=[0,0,0])

image = image.reshape((700*400))

pipe.stdout.flush()
pipe.terminate()

image = image.astype(np.float32)

c = conv.conv()

output = c.convolutional_neural_network(image)
batch_size = 128

x = tf.placeholder('float', [None, 700*400])
y = tf.placeholder('float')
keep_prob = tf.placeholder(tf.float32)

def train_neural_network(x):
    prediction = c.convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


print(output.shape)

