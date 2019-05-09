import os
import numpy as np
import tensorflow as tf
import glob


def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def create_mnist_tfr(data_dir, tfrecord_dir='mnist'):
    resolution_log2 = int(np.log2(32))

    # reading raw images
    import gzip
    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 1, 28, 28)

    # 28x28 -> 32x32
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_values=0)

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/mnist%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = np.shape(images)[0]
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = images[order[idx]]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())


def create_cifar10_tfr(data_dir, tfrecord_dir='cifar10'):
    resolution_log2 = int(np.log2(32))

    # reading raw images
    import pickle
    images = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(data_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(data['labels'])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/cifar10%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = np.shape(images)[0]
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = images[order[idx]]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

    with open(os.path.join('datasets', tfrecord_dir, 'cifar10-rxx.labels'), 'wb') as f:
        np.save(f, onehot[order].astype(np.float32))


def create_portrait_tfr(data_dir, tfrecord_dir='portrait'):
    resolution_log2 = int(np.log2(256))

    # reading raw images
    import PIL.Image
    glob_pattern = os.path.join(data_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/portrait%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = len(image_filenames)
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]).resize((256,256))).transpose(2, 0, 1)
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def data_iterator(dataset, lod_in, batch_size, resolution_log2, use_labels=False):
    
    tfrecord_dir = os.path.join('datasets', dataset)
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    images = []
    count = 0

    if use_labels != False:
        np_labels = np.load(os.path.join('datasets', dataset, '%s-rxx.labels' % dataset))
        labels = []

        while True:
            for i, record in enumerate(tf.python_io.tf_record_iterator(os.path.join(tfrecord_dir, '%s%02d.tfrecords' % (dataset, int(resolution_log2-np.floor(lod_in)))), tfr_opt)):
                count += 1
                images.append(parse_tfrecord_np(record))
                labels.append(np_labels[i])
                if count >= batch_size:
                    yield np.asarray(images), np.asarray(labels)
                    count = 0
                    images = []
                    labels = []

    else:
        while True:
            for i, record in enumerate(tf.python_io.tf_record_iterator(os.path.join(tfrecord_dir, '%s%02d.tfrecords' % (dataset, int(resolution_log2-np.floor(lod_in)))), tfr_opt)):
                count += 1
                images.append(parse_tfrecord_np(record))
                if count >= batch_size:
                    yield np.asarray(images)
                    count = 0
                    images = []

if __name__ == "__main__":

    tf.app.flags.DEFINE_string('dataset', 'mnist', 'Dataset to use')
    tf.app.flags.DEFINE_string('path', None, 'raw image path')
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.dataset == 'mnist':
        create_mnist_tfr(data_dir=FLAGS.path)
    elif FLAGS.dataset == 'cifar10':
        create_cifar10_tfr(data_dir=FLAGS.path)
    elif FLAGS.dataset == 'portrait':
        create_portrait_tfr(data_dir=FLAGS.path)