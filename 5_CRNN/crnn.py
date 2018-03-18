import numpy as np
import os
import conv
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
        
        self.train_count = len(X_train)  # 7352 training series
        # self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = 5#len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 100
        self.batch_size = 90

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 32#32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output'], config.W, config.biases
  

    

if __name__ == "__main__":
    train_x = np.load('./data/train_x.npz')['a']
    train_y = np.load('./data/train_y.npz')['a']

    x = tf.placeholder('float', [None, 700*400])
    y = tf.placeholder('float',[None, 10]) 
    c = conv.conv()

    prediction = c.convolutional_neural_network(x)
<<<<<<< HEAD
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
=======
    cost_cnn = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost_cnn)
>>>>>>> e73c6de2c8fa6e64e08d15e416f43f568eee7967
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train_x = x; train_y = y
        for epoch in range(hm_epochs):
            _, c = sess.run([optimizer, cost_cnn], feed_dict={x: train_x, y: train_y})
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',c)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print(accuracy.eval())
        # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    ###################################################################################################
    '''
    config = Config(train_x)

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])    
    
    pred_Y, W, B = LSTM_Network(X, config)
    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out, weight_trained, biases_trained = sess.run(
            [pred_Y, accuracy, cost, W, B],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
        
    
    
        print("training iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)
    
    '''
    '''save weights and biases'''
    '''
    np.save('./weights/weight_hidden1',weight_trained['hidden'])
    np.save('./weights/weight_output1',weight_trained['output'])
    np.save('./weights/biases_hidden1',biases_trained['hidden'])
    np.save('./weights/biases_output1',biases_trained['output'])

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    '''