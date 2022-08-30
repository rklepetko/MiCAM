import tensorflow as tf
FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_integer('window_size', 3, "In case of 3d CNN, provide a window size. window_size = 1 is the same as 2d CNN.")
tf.compat.v1.app.flags.DEFINE_integer('height', 150, "Height of a single sample.")
tf.compat.v1.app.flags.DEFINE_integer('width', 75, "Width of a single sample.")
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 64, "Number of sample in a single batch.")
tf.compat.v1.app.flags.DEFINE_integer('num_epochs', 20, "Number of training epochs.")
tf.compat.v1.app.flags.DEFINE_string('data_dir', '~/tfrecords/*', "TFRecords directory.")
tf.compat.v1.app.flags.DEFINE_string('data_file', '~/tfrecords/*', "TFRecords file.")
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 1e-5, "Optimizer learning rate.")
##tf.compat.v1.app.flags.DEFINE_string('log_dir', 'log/e1', "Location of saving logs")
tf.compat.v1.app.flags.DEFINE_string('cnn_model', 'resnet50', "CNN model to use")
tf.compat.v1.app.flags.DEFINE_float('dropout', 0.5, "No dropout is set with 0 (zero).")
tf.compat.v1.app.flags.DEFINE_integer('classes', 2, "Use for more than 2 classes")
tf.compat.v1.app.flags.DEFINE_boolean('batch_normalization', False, "Enable batch normalization")