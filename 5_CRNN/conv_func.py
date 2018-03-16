import tensorflow as tf
import numpy as np

class conv():
    def __init__(self):
        self.n_classes = 10
        self.keep_rate = 0.8

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    def maxpool2d(self,x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    def convolutional_neural_network(self,x):
        


        weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
                'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                'W_fc':tf.Variable(tf.random_normal([25*31*128,1024])),
                'out':tf.Variable(tf.random_normal([1024, self.n_classes]))}

        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                'b_conv2':tf.Variable(tf.random_normal([64])),
                'b_fc':tf.Variable(tf.random_normal([1024])),
                'out':tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.reshape(x, shape=[-1, 198, 124, 1])

        conv1 = tf.nn.relu(self.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self.maxpool2d(conv1)
        
        conv2 = tf.nn.relu(self.conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self.maxpool2d(conv2)

        fc = tf.reshape(conv2,[-1, 25*31*128])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out'])+biases['out']

        return output

    

    '''
    def conv2d(self,inputdata, out_channel, kernel_size, padding='SAME', stride=1, w_init=None, b_init=None,
                nl=tf.identity, split=1, use_bias=True, data_format='NHWC', name=None):
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                            for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name=name)

        return ret
    

    def relu(self, inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    
    def sigmoid(self, inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    
    def maxpooling(self, inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                            data_format=data_format, name=name)

    def feature_extraction(self,inputdata):
        conv1 = self.conv2d(inputdata=inputdata, out_channel=64, kernel_size=3, stride=1, use_bias=False, name='conv1') 
        relu1 = self.relu(inputdata=conv1)
        max_pool1 = self.maxpooling(inputdata=relu1, kernel_size=2, stride=2)
        conv2 = self.conv2d(inputdata=max_pool1, out_channel=128, kernel_size=3, stride=1, use_bias=False, name='conv2') 
        relu2 = self.relu(inputdata=conv2)
        max_pool2 = self.maxpooling(inputdata=relu2, kernel_size=2, stride=2)
        conv3 = self.conv2d(inputdata=max_pool2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3')  # batch*8*25*256 #66*27*256
        relu3 = self.relu(conv3) # batch*8*25*256
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*25*256 #66*27*256
        relu4 = self.relu(conv4)  # batch*8*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[4, 1], stride=[4, 1], padding='VALID')  # batch*4*25*256 #16*27*256
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*25*512 #16*27*512 
        relu5 = self.relu(conv5)

        return relu5
    '''
    