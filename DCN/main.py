import os
from datetime import datetime
import time
import tensorflow as tf
import json
import cv2
import numpy as np

class DataReader(object):

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, examples = reader.read(filename_queue)
        features = tf.parse_single_example(
            examples,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image-0.5) * 2
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)
        return image, length, digits

    @staticmethod
    def build_batch(tfrecords_file, num_examples, batch_size, shuffled):
        assert tf.gfile.Exists(tfrecords_file), '%s not found' % tfrecords_file

        filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=None)
        image, length, digits = DataReader._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],
                                                                             batch_size=batch_size,
                                                                             num_threads=2,
                                                                             capacity=min_queue_examples + 3 * batch_size,
                                                                             min_after_dequeue=min_queue_examples)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size)
        return image_batch, length_batch, digits_batch



class Evaluator(object):
    def __init__(self):
        pass

    def evaluate(self, checkpoint_file, tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples // batch_size
        needs_include_length = False

        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = DataReader.build_batch(tfrecords_file,
                                                                         num_examples=num_examples,
                                                                         batch_size=batch_size,
                                                                         shuffled=False)
            length_logits, digits_logits = Model.inference(image_batch, drop_rate=0.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)

            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, checkpoint_file)

                for _ in range(int(num_batches)):
                    sess.run(update_accuracy)

                accuracy_val = sess.run(accuracy)
                coord.request_stop()
                coord.join(threads)

        return accuracy_val


def conv_layer(name, inpt, filters, ksize, stride, drop_rate):
    with tf.variable_scope(name):
            conv = tf.layers.conv2d(inpt, filters=filters, kernel_size=[ksize, ksize], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=stride, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
    return dropout

def ce_loss(labels, logits):
    return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))

class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        l = conv_layer('conv1', x, 48, 5, 2, drop_rate)
        l = conv_layer('conv2', l, 64, 5, 1, drop_rate)
        l = conv_layer('conv3', l, 128, 5, 2, drop_rate)
        l = conv_layer('conv4', l, 160, 5, 1, drop_rate)
        l = conv_layer('conv5', l, 192, 5, 2, drop_rate)
        l = conv_layer('conv6', l, 192, 5, 1, drop_rate)
        l = conv_layer('conv7', l, 192, 5, 2, drop_rate)
        l = conv_layer('conv8', l, 192, 5, 1, drop_rate)
        l = tf.reshape(l, [-1, 4 * 4 * 192])

        with tf.variable_scope('dense9'):
            l = tf.layers.dense(l, units=3072, activation=tf.nn.relu)

        with tf.variable_scope('dense10'):
            l = tf.layers.dense(l, units=3072, activation=tf.nn.relu)

        with tf.variable_scope('length'):
            length = tf.layers.dense(l, units=7)

        with tf.variable_scope('d1'):
            digit1 = tf.layers.dense(l, units=11)

        with tf.variable_scope('d2'):
            dense = tf.layers.dense(l, units=11)
            digit2 = dense

        with tf.variable_scope('d3'):
            digit3 = tf.layers.dense(l, units=11)

        with tf.variable_scope('d4'):
            digit4 = tf.layers.dense(l, units=11)

        with tf.variable_scope('d5'):
            digit5 = tf.layers.dense(l, units=11)

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits


    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        loss =  ce_loss(labels=length_labels, logits=length_logits)
        for i in range(5):
            loss += ce_loss(labels=digits_labels[:, i], logits=digits_logits[:, i, :])
        return loss

def _train(path_train, num_train_imgs, path_val, num_val_imgs,
           path_log_dir, path_checkpoint_file, train_config):
    batch_size = train_config['batch_size']
    initial_patience = train_config['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = DataReader.build_batch(path_train,
                                                                     num_examples=num_train_imgs,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        length_logtis, digits_logits = Model.inference(image_batch, drop_rate=0.2)
        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(train_config['learning_rate'], global_step=global_step,
                                                   decay_steps=train_config['decay_steps'], decay_rate=train_config['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)


        with tf.Session() as sess:
            evaluator = Evaluator()

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_checkpoint_file), \
                    '%s not found' % path_checkpoint_file
                saver.restore(sess, path_checkpoint_file)
                print ('Model restored from file: %s' % path_checkpoint_file)

            print( 'Start training...')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, global_step_val, learning_rate_val = sess.run([train_op, loss, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print( '%s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec))

                if global_step_val % num_steps_to_check != 0:
                    continue

                print ('Evaluating...')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_val,
                                              num_val_imgs,
                                              global_step_val)
                print ('accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)


def train(FLAGS):
    path_train = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_val = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_tfrecords_json = os.path.join(FLAGS.data_dir, 'meta.json')
    path_log_dir = FLAGS.ckpt_dir
    path_checkpoint_file = FLAGS.restore_checkpoint
    train_config = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }

    
    with open(path_tfrecords_json, 'r') as f:
            data_disc = json.load(f)
            num_train_imgs = data_disc['num_examples']['train']
            num_val_imgs = data_disc['num_examples']['val']

    _train(path_train, 
           num_train_imgs,
           path_val, 
           num_val_imgs,
           path_log_dir, 
           path_checkpoint_file,
           train_config)


def _eval(path_checkpoint_dir, path_eval, num_eval_imgs, path_eval_log_dir):
    evaluator = Evaluator()

    checkpoint_paths = tf.train.get_checkpoint_state(path_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluator.evaluate(path_to_checkpoint, path_eval, num_eval_imgs, global_step_val)
        print ('Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint, path_eval, accuracy))


def eval(FLAGS):
    path_val = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_test = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_tfrecords_json = os.path.join(FLAGS.data_dir, 'meta.json')
    path_checkpoint_dir = FLAGS.checkpoint_dir

    path_val_log_dir = os.path.join(FLAGS.eval_logdir, 'val')
    path_test_log_dir = os.path.join(FLAGS.eval_logdir, 'test')

    with open(path_tfrecords_json, 'r') as f:
        data_disc = json.load(f)
        num_val_imgs = data_disc['num_examples']['val']
        num_test_imgs = data_disc['num_examples']['test']

    _eval(path_checkpoint_dir, path_val, num_val_imgs, path_val_log_dir)
    _eval(path_checkpoint_dir, path_test, num_test_imgs, path_test_log_dir)

def test(FLAGS, img_path):
    image = cv2.imread(img_path).astype(np.float32)
    image = (image - 0.5) * 2
    image = cv2.resize(image, (54, 54))

    images = tf.placeholder(shape=[1, 54,54,3], dtype=tf.float32)
    length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
    length_pred_op = tf.argmax(length_logits, axis=1)
    digits_pred_op= tf.argmax(digits_logits, axis=2)
    digits_str_pred_op = tf.reduce_join(tf.as_string(digits_pred_op), axis=1)

    with tf.Session() as sess:
        restorer = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir):
            latest_check_point = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
            restorer.restore(sess, latest_check_point)
        else:
            raise '[ERROR] Checkpoint file not found'

        length, digits = sess.run([length_pred_op, digits_str_pred_op], feed_dict = {images:[image]})
        length = length[0]
        digits = digits[0]
        print(length, digits)


def get_DCN(sess, flags):
    batch_size = 1
    ckpt_paths = tf.train.get_checkpoint_state(flags.DCN_checkpoint_dir).all_model_checkpoint_paths
    ckpt_file =  ckpt_paths[-1] 

    images = tf.placeholder(shape=[1, 54,54,3], dtype=tf.float32)
    length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
    length_pred_op = tf.argmax(length_logits, axis=1)
    digits_pred_op= tf.argmax(digits_logits, axis=2)
    digits_str_pred_op = tf.reduce_join(tf.as_string(digits_pred_op), axis=1)

    restorer = tf.train.Saver()
    restorer.restore(sess, ckpt_file)

    return [images, length_pred_op, digits_str_pred_op]

def expand_box(box):
    x1, y1, x2, y2 = box
    xsize = x2 - x1
    ysize = y2 - y1
    return [x1 - xsize//4, y1 - ysize//4, x2 + xsize//4, y2 + ysize//4]

        
def crop_img(img, box):
    height, width, __= img.shape
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if y1 < y2 and x1 < x2:
        patch = img[y1:y2, x1:x2, :]
    else:
        patch = img
    return patch

def recognize(sess, DCN, img, boxes):
    images, length_pred_op, digits_str_pred_op = DCN
    length_list, digits_list = [], []
    for box in boxes:
        box = expand_box(box)
        img = cv2.resize(crop_img(img, box), (54, 54))
        length, digits = sess.run([length_pred_op, digits_str_pred_op], feed_dict = {images:[img]})
        length = length[0]
        digits = digits[0]
        #print(length, digits)
        length_list.append(length)
        digits_list.append(digits)
    return length_list, digits_list


if __name__ == '__main__':
    mode = 'train'
    tf.app.flags.DEFINE_string('data_dir', '../data/SVHN', 'directory of TFRecords files')
    tf.app.flags.DEFINE_string('ckpt_dir', './ckpt', 'directory to write logs')
    tf.app.flags.DEFINE_string('restore_checkpoint', None,
                               'Path to restore checkpoint, e.g. ./logs/train/model.ckpt-100')
    tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
    tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
    tf.app.flags.DEFINE_integer('patience', 100, '')
    tf.app.flags.DEFINE_integer('decay_steps', 10000, 'learning rate decay steps')
    tf.app.flags.DEFINE_float('decay_rate', 0.9, 'learning rate decay rate')
    FLAGS = tf.app.flags.FLAGS
    if mode == 'train':
        train(FLAGS)
    elif mode == 'val':
        val(FLAGS)
    else:
        img_path = '../data/imgs/2.jpg'
        test(FLAGS, img_path)
