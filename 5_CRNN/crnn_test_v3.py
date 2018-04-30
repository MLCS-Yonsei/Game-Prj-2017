import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

model_dir = '/home/jehyunpark/Downloads/crnn/results/'
image_path = '/home/jehyunpark/Downloads/crnn/images/boxing/'


BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
# BOTTLENECK_TENSOR_SIZE = 2048
# MODEL_INPUT_WIDTH = 299
# MODEL_INPUT_HEIGHT = 299
# MODEL_INPUT_DEPTH = 3
i = 0

filenames = sorted(os.listdir(image_path), key = lambda a:a[6:11])[20:30]

class Config(object):

  def __init__(self):
    # Input data
    W_h=np.load('./weights/weight_hidden.npy')
    W_o=np.load('./weights/weight_output.npy')
    B_h=np.load('./weights/biases_hidden.npy')
    B_o=np.load('./weights/biases_output.npy')

    # Training
    self.learning_rate = 0.0025
    self.lambda_loss_amount = 0.0015
    self.training_epochs = 200
    self.batch_size = 90
    self.n_steps = 10  # 128 time_steps per series


    # LSTM structure
    self.n_inputs = 2048  # Features count is of 9: 3 * 3D sensors features over time
    self.n_hidden = 32#32  # nb of neurons inside the neural network
    self.n_classes = 6  # Final output classes
    self.W = {
        'hidden': tf.Variable(W_h),
        'output': tf.Variable(W_o)
    }
    self.biases = {
        'hidden': tf.Variable(B_h),
        'output': tf.Variable(B_o)
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
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(
        model_dir, 'output_graph.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.
  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.
  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

if __name__ == "__main__":
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
            create_inception_graph())

    with tf.Session(graph=graph) as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        for filename in filenames:
            full_filename = os.path.join(image_path,filename)
            if i == 0:
                jpeg_data = gfile.FastGFile(full_filename, 'rb').read()
                frames = run_bottleneck_on_image(sess, jpeg_data, jpeg_data_tensor, bottleneck_tensor)[np.newaxis,:]
                i +=1
            elif len(frames) < 10:
                jpeg_data = gfile.FastGFile(full_filename, 'rb').read()
                frames = np.concatenate((frames, run_bottleneck_on_image(sess, jpeg_data, jpeg_data_tensor, bottleneck_tensor)[np.newaxis,:]), axis = 0)
    frames = frames[np.newaxis,:,:]
    config = Config()

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    pred_Y = LSTM_Network(X, config)    
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1):
        pred_out = sess.run(
            [pred_Y],
            feed_dict={
                X: frames
            }
        )

    a = np.argmax(pred_out)
    print(a)
    # if a == 0:
    #     result = 'walking'
    # elif a == 1:
    #     result = 'boxing'
    # elif a == 2:
    #     result = 'handwaving'
    # elif a == 3:
    #     result = 'jogging'
    # elif a == 4:
    #     result = 'running'
    # elif a == 5:
    #     result = 'handclapping'
    # print(result, pred_out)
    
    
