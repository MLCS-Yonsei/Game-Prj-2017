import numpy as np
import os
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train):
        # Input data
        self.train_count = len(X_train[0])  # 7352 training series
        # self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = 10#len(X_train[0])  # 128 time_steps per series
        self.img_h = 720
        self.img_w = 1262

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 10#100
        self.batch_size = 5#90

        # LSTM structure
        self.n_inputs = 10#len(X_train[0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 10  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.b = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.weights = {'W_conv1':tf.Variable(tf.random_normal([50,50,1,16])),#[5,5,1,32]
                'W_conv2':tf.Variable(tf.random_normal([5,5,16,16])),#[5,5,32,64]
                'W_fc':tf.Variable(tf.random_normal([7*4*16,256])),#[35*125*256]
                'out':tf.Variable(tf.random_normal([256, self.n_classes]))}

        self.biases = {'b_conv1':tf.Variable(tf.random_normal([16])),
                'b_conv2':tf.Variable(tf.random_normal([16])),
                'b_fc':tf.Variable(tf.random_normal([256])),
                'out':tf.Variable(tf.random_normal([self.n_classes]))}
        self.keep_rate = 0.8

def CRNN(_X, _Y, config):
    _X = tf.reshape(_X, shape=[-1, config.img_h, config.img_w, 1])
    _X = tf.cast(_X, tf.float32)
    conv1 = tf.nn.relu(tf.nn.conv2d(_X, config.weights['W_conv1'], strides=[1,10,10,1], padding='SAME') + config.biases['b_conv1'])
    print(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME')
    print(conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, config.weights['W_conv2'], strides=[1,5,5,1], padding='SAME') + config.biases['b_conv2'])
    print(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    print(conv2)

    fc = tf.reshape(conv2,[-1, 7*4*16])#[35*125*256]
    print(fc)
    fc = tf.nn.relu(tf.matmul(fc, config.weights['W_fc']) + config.biases['b_fc'])
    print(fc)
    fc = tf.nn.dropout(fc, config.keep_rate)

    out = tf.matmul(fc, config.weights['out']) + config.biases['out']
    print(out)
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    out = tf.reshape(out, [-1,config.n_steps,config.n_classes])
    print(out)
    out = tf.transpose(out, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    print(out)
    out = tf.reshape(out, [-1, config.n_inputs])
    print(out)
    # out = tf.cast(out, tf.float32)
    # new shape: (n_steps*batch_size, n_input)
    # Linear activation
    out = tf.nn.relu(tf.matmul(out, config.W['hidden']) + config.b['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    out = tf.split(out, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, out, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.b['output'], _Y, config.W, config.b, config.weights, config.biases

if __name__ == "__main__":
    train_x = np.load('/home/jehyunpark/data/train_x.npz')['a']
    train_x[:int(0.7*len(train_x)),:]
    test_x[int(0.7*len(train_x)):,:]
    train_x = np.reshape(train_x,[-1,10,1262*720])
    
    train_y = np.load('/home/jehyunpark/data/train_y.npz')['a']
    train_y[:int(0.7*len(train_y)),:]
    test_y[int(0.7*len(train_y)):,:]

    print('data loading completed')
    
    config = Config(train_x)
    X = tf.placeholder(tf.float32, [None,10, config.img_h*config.img_w])
    Y = tf.placeholder(tf.float32,[None, config.n_classes])
    
    # a,b,c,d,e,f = CRNN(train_x,train_y,config)
    # print(b.shape)
    
    prediction, label, W, B, weights, biases = CRNN(X, Y, config)
    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction) ) + l2
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    cfg = tf.ConfigProto()
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.85
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config= cfg)

    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        best_accuracy = 0.0
        # Start training for each batch and loop epochs
        for i in range(config.training_epochs):
            
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                range(config.batch_size, config.train_count + 1, config.batch_size)):
                sess.run(optimizer, feed_dict={X: train_x[start:end],
                                            Y: train_y[start:end]})
            
            # Test completely at every epoch: calculate accuracy
            pred_out, accuracy_out, loss_out, W_, B_, weights_, biases_ = sess.run(
                [prediction, accuracy, cost, W, B, weights, biases], feed_dict={X: test_x, Y: test_y}
            )

            print("training iter: {},".format(i) +
                " test accuracy : {},".format(accuracy_out) +
                " loss : {}".format(loss_out))
            best_accuracy = max(best_accuracy, accuracy_out)  

    np.savez_compressed('./data/W_hidden',a=W_['hidden'])
    np.savez_compressed('./data/W_output',a=W_['output'])
    np.savez_compressed('./data/b_hidden',a=B_['hidden'])
    np.savez_compressed('./data/b_output',a=B_['output'])
    np.savez_compressed('./data/W_conv1',a=weights_['W_conv1'])
    np.savez_compressed('./data/W_conv2',a=weights_['W_conv2'])
    np.savez_compressed('./data/W_fc',a=weights_['W_fc'])
    np.savez_compressed('./data/W_out',a=weights_['out'])
    np.savez_compressed('./data/b_conv1',a=biases_['b_conv1'])
    np.savez_compressed('./data/b_conv2',a=biases_['b_conv2'])
    np.savez_compressed('./data/b_fc',a=biases_['b_fc'])
    np.savez_compressed('./data/b_out',a=biases_['out'])

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    
    sess.close()
    
    
    
 