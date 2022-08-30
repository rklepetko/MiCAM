import gzip
import os
import numpy
from six.moves import urllib
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

params = {}
params['download_data_location'] = '../MNIST/data'
params['tfrecord_location'] = '../MNIST/75x75/tfrecords'


def download(directory, filename):
  """Download a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  temp_file_name, _ = urllib.request.urlretrieve(url)
  tf.gfile.Copy(temp_file_name, filepath)
  with tf.gfile.GFile(filepath) as f:
      size = f.size()
  print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """

  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """

  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))

    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)

    return labels

def load_dataset(directory, images_file, labels_file):
  """Download and parse MNIST dataset."""

  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  with tf.gfile.Open(images_file, 'rb') as f:
    images = extract_images(f)

  with tf.gfile.Open(labels_file, 'rb') as f:
    labels = extract_labels(f)

  return images, labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(val):
  return tf.train.Feature(float_list=tf.train.FloatList(value=val))

directory = params['download_data_location']
validation_size=5000
train_images, train_labels = load_dataset(directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
test_images, test_labels = load_dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte') 
images = numpy.concatenate((train_images,test_images),axis=0)
labels = numpy.concatenate((train_labels,test_labels),axis=0)

image_front = []
label_front = []
num = 0
for i in range(images.shape[0]):
  if num == int(labels[i]):
    #print(num)
    #print(labels[i])
    image_front.append(images[i])
    label_front.append(labels[i])
    num = num + 1
  if num >= 10:
    continue
     
images = numpy.concatenate((image_front,images),axis=0)
labels = numpy.concatenate((label_front,labels),axis=0)

#print(images.shape)
images = tf.image.resize_images(images,[75,75]).numpy()


# plot MNIST images, set to true
if (false):
    for i in range(images.shape[0]):
        plt.imshow(images[i], cmap='gray')
        plt.title(str(labels[i]))
#        plt.show()
        plt.savefig("./images/"+str(labels[i])+"-"+str(i)+".jpg", dpi=600, orientation='portrait')
        plt.close()

name = "mnist.tfrecord"
filename = os.path.join(params['tfrecord_location'], name)
tfrecord_writer = tf.python_io.TFRecordWriter(filename)

images.shape[0]
num_examples = images.shape[0]

rows = images.shape[1]
cols = images.shape[2]
depth = images.shape[3]

print("MINMAX")
print(numpy.min(images))
print(numpy.max(images))
for index in range(num_examples):

  # 1. Convert your data into tf.train.Feature
  #image_raw = images[index].tobytes()
  image_flt = numpy.squeeze(numpy.array(images[index],dtype="float64"))
  image_flt = image_flt.reshape(rows*cols)
  feature = {
    'sample': _floats_feature(image_flt),
    'label': _int64_feature(int(labels[index]))
  }

  # 2. Create a tf.train.Features
  features = tf.train.Features(feature=feature)

  # 3. Createan example protocol
  example = tf.train.Example(features=features)

  # 4. Serialize the Example to string
  example_to_string = example.SerializeToString()

  # 5. Write to TFRecord
  tfrecord_writer.write(example_to_string)

print({
    'height': _int64_feature(rows),
    'width': _int64_feature(cols),
    'depth': _int64_feature(depth),
    'length': num_examples})
  
