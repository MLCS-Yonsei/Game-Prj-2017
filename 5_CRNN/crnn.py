import numpy as np
import subprocess as sp
import os
import json
import conv
import tensorflow as tf
import cv2

train_x = np.load('train_x.npz')['a']
train_y = np.load('train_y.npz')['a']
# c = conv.conv()
# output = c.convolutional_neural_network(image)
# print(output.shape)

# keep_prob = tf.placeholder(tf.float32)

'''
def train_neural_network(x,y):
    x = tf.placeholder('float', [None, 700*400])
    y = tf.placeholder('float') 

    prediction = c.convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_x = x; train_y = y
        for epoch in range(hm_epochs):
            epoch_loss = 0
             
            _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
'''


