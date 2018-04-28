import tensorflow as tf
import numpy as np

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 1000000
        self.batch_size = 100

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
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
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
    lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=0.7)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob=0.7)
    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_3, output_keep_prob=0.7)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output'], config.W, config.biases


if __name__ == "__main__":
  #load data, training set dimension is 1260*5*66, test set dimension is 540*5*66
    X_train=np.load('/home/jehyunpark/Downloads/crnn/data/train_x.npz')['a']
    X_train=np.reshape(X_train,(-1,10,2048))
    # np.take(X_train,np.random.permutation(10),axis=1,out=X_train)
    y_train=np.load('/home/jehyunpark/Downloads/crnn/data/train_y.npz')['a']
    X_test=np.load('/home/jehyunpark/Downloads/crnn/data/test_x.npz')['a']
    X_test=np.reshape(X_test,(-1,10,2048))
    y_test=np.load('/home/jehyunpark/Downloads/crnn/data/test_y.npz')['a']    

    config = Config(X_train, X_test)

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

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
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
        # sess.run(optimizer, feed_dict={X: X_train,
        #                                    Y: y_train})
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
        if best_accuracy == accuracy_out:
            np.save('./weights/weight_hidden3',weight_trained['hidden'])
            np.save('./weights/weight_output3',weight_trained['output'])
            np.save('./weights/biases_hidden3',biases_trained['hidden'])
            np.save('./weights/biases_output3',biases_trained['output'])
        
    
    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    
