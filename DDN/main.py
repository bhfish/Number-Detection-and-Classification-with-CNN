import time
import os
import tensorflow as tf
import glob
import cv2
import random
import numpy as np
import argparse
import xml.dom.minidom



def conv_block(inpt, n_filters, training, flags, name, pool=True, activation=tf.nn.relu):

    l = inpt
    with tf.variable_scope(name):
        for idx, n_filter in enumerate(n_filters):
            l = tf.layers.conv2d(
                l,
                n_filter, 
                (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(idx + 1))
            l = tf.layers.batch_normalization(l, training=training, name="bn_{}".format(idx + 1))
            l = activation(l, name="relu_{}".format(idx + 1))

        if pool is False:
            return l
        else:
            pool = tf.layers.max_pooling2d(l, (2, 2), strides=(2, 2), name="pool")
            return l, pool

def conv_dilate_block(inpt, n_filters, dilations, training, flags, name, pool=True, activation=tf.nn.relu):

    l = inpt
    with tf.variable_scope(name):
        for idx, n_filter in enumerate(n_filters):
            l = tf.layers.conv2d(
                l,
                n_filter, 
                (3, 3),
                dilation_rate=(dilations[idx], dilations[idx]),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(idx + 1))
            l = tf.layers.batch_normalization(l, training=training, name="bn_{}".format(idx + 1))
            l = activation(l, name="relu_{}".format(idx + 1))

        if pool is False:
            return l
        else:
            l = tf.layers.max_pooling2d(l, (2, 2), strides=(2, 2), name="pool")
            return net, pool


def upconv_block(inpt1, inpt2, n_filter, flags, name):
    with tf.variable_scope(name):
        up_conv = tf.layers.conv2d_transpose(
                  inpt1,
                  filters=n_filter,
                  kernel_size=2,
                  strides=2,
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                  name="upconv")
        concat = tf.concat([up_conv, inpt2], axis=-1, name="concat")
        return concat

def unet(X, training, flags=None):

    net = X / 127.5 - 1
    conv1, down1 = conv_block(net, [8, 8], training, flags, name='down1')
    conv2, down2 = conv_block(down1, [16, 16], training, flags, name='down2')
    conv3, down3 = conv_block(down2, [32, 32], training, flags, name='down3')
    conv4, down4 = conv_block(down3, [64, 64], training, flags, name='down4')
    conv5, down5 = conv_block(down4, [128, 128], training, flags, name='down5')

    dilated = conv_dilate_block(down5, [256, 256, 256, 256], [2, 4, 8, 16], training, flags, name='down6', pool=False)
    up5 = upconv_block(dilated, conv5, 128, flags, name='dilate')
    up5 = conv_block(up5, [128, 128], training, flags, name='up5', pool=False)

    up4 = upconv_block(up5, conv4, 64, flags, name='up4')
    up4 = conv_block(up4, [64, 64], training, flags, name='up4', pool=False)

    up3 = upconv_block(up4, conv3, 32, flags, name='up3')
    up3 = conv_block(up3, [32, 32], training, flags, name='up3', pool=False)

    up2 = upconv_block(up3, conv2, 16, flags, name='up2')
    up2 = conv_block(up2, [16, 16], training, flags, name='up2', pool=False)

    up1 = upconv_block(up2, conv1, 8, flags, name='up1')
    up1 = conv_block(up1, [8, 8], training, flags, name='up1', pool=False)
    
    y = tf.layers.conv2d(up1, 1, (1, 1), name='y_op', activation=tf.nn.sigmoid, padding='same')
    box = tf.layers.conv2d(up1, 3, (1, 1), name='box_op', activation=None, padding='same')
    return y, box


def get_iou(y_pred, y_true):
    pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    true = tf.reshape(y_true, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(pred * true, axis=1) + 1e-4
    union = tf.reduce_sum(tf.maximum(pred, true), axis=1) + 1e-4

    return tf.reduce_mean(intersection / union)


def get_train_op(y_pred, box_pred, y_true, box_true):

    loss = -get_iou(y_pred, y_true)
    loss += l2_alpha * tf.reduce_mean(tf.square(box_pred-box_true) * y_true)/ (tf.reduce_mean(y_true) + 1e-4)


    global_step = tf.train.get_or_create_global_step()
    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    return optim.minimize(loss, global_step=global_step)

def assign(label_map, box, h ,w):
    x1, y1, x2, y2 = box
    hh = np.tile(np.reshape(range(h), [h,1]), [1, w])
    ww = np.tile(np.reshape(range(w), [1,w]), [h, 1])
    x, y = (x1+x2)/2., (y1+y2)/2.
    label_map[y1:y2, x1:x2, 1:3]  = np.stack([(x - ww)/w, (y - hh)/h], axis=2)[y1:y2, x1:x2, :]
    label_map[y1:y2, x1:x2, 0] = 1.
    label_map[y1:y2, x1:x2, 3] = (x2-x1)/(1.*w)


# assume there is only one box
def gen_label_single_box(box, h, w):
    label_map = np.zeros(shape=[h,w,4])
    if box is not None:
        assign(label_map, box, h , w)
    return label_map

def gen_label(avoid_boxes, height, width):
    h, w = height, width
    label_map = np.zeros(shape=[h,w,4])
    for box in avoid_boxes:
        assign(label_map, box, h , w)
    return label_map

def parse_xml(xml_file):
    if os.path.exists(xml_file):
      basename = os.path.basename(xml_file)
      doc = xml.dom.minidom.parse(xml_file);
      filename = doc.getElementsByTagName("filename")[0].firstChild.data
      assert filename[:-4] == basename[:-4], 'filename in xml not consistent'
      x1 = int(doc.getElementsByTagName("xmin")[0].firstChild.data)
      x2 = int(doc.getElementsByTagName("xmax")[0].firstChild.data)
      y1 = int(doc.getElementsByTagName("ymin")[0].firstChild.data)
      y2 = int(doc.getElementsByTagName("ymax")[0].firstChild.data)
      return [x1, y1, x2, y2]
    else:
      return None

def augment_data(x, y, box):
    for idx in range(len(x)):
        img, label, bbox = x[idx], y[idx], box[idx]
        height, width, _ = img.shape
        # scale
        r = np.random.rand(1)[0] 
        scale = r * 0.2 + 0.8
        scale = min(scale, 1024./max(height, width))
        #scale = max(scale, 64./min(height, width))
        height, width = int(height * scale), int(width * scale)
        height = (height//32) * 32
        width  = (width//32) * 32
        img = cv2.resize(img, (width, height))
        label = cv2.resize(label, (width, height))
        bbox = cv2.resize(bbox, (width, height))
        # rotation
        angle = random.randint(-10, 10)
        rotation_mat = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
        img = cv2.warpAffine(img, rotation_mat, (width, height))
        label = cv2.warpAffine(label, rotation_mat, (width, height))
        bbox = cv2.warpAffine(bbox, rotation_mat, (width, height))
        label = np.expand_dims(label, axis=3)
        x[idx], y[idx], box[idx] = img, label, bbox

def get_data_batch(t_data, n_data, batch_size):
    idxes = random.sample(range(n_data), batch_size)
    img_pathss = [t_data['img'][idx] for idx in idxes]
    label_paths = [t_data['label']+'/'+os.path.basename(f)[:-4]+'.xml' for f in img_pathss]
    x, y, box = [] ,[], []
    for f1, f2 in zip(img_pathss, label_paths):
       img = cv2.imread(f1)
       height, width, __ = img.shape
       labels = parse_xml(f2)
       label_map = gen_label_single_box(labels, height, width) 
       mask = label_map[:,:,0:1]
       bbox = label_map[:,:,1:]
       x.append(img), y.append(mask), box.append(bbox)
    augment_data(x, y, box)
    return np.asarray(x), np.asarray(y), np.asarray(box)

def get_data(data_dir):
    data = {}
    data['img'] = glob.glob(data_dir+'/*.jpg')
    data['label'] = data_dir
    n_data = len(data['img'])
    return data, n_data

def train(flags):
    train_data, n_train = get_data(train_data_dir)
    test_data, n_test = get_data(test_data_dir)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="y")
    box_true = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="box_true")
    mode = tf.placeholder(tf.bool, name="mode")

    X = tf.image.random_brightness(X, 0.7)
    X = tf.image.random_hue(X, 0.3)
    pred, box_pred = unet(X, mode, flags)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = get_train_op(pred, box_pred, y, box_true)

    get_iouop = get_iou(pred, y)
    box_loss_op = tf.reduce_mean(tf.square(box_pred-box_true) * y) / (tf.reduce_mean(y) + 1e-4)
    batch_size = flags.batch_size

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.exists(flags.checkpoint_dir) and tf.train.checkpoint_exists(
                flags.checkpoint_dir):
            latest_check_point = tf.train.latest_checkpoint(flags.checkpoint_dir)
            saver.restore(sess, latest_check_point)

        else:
            try:
                os.rmdir(flags.checkpoint_dir)
            except FileNotFoundError:
                pass
            os.mkdir(flags.checkpoint_dir)

        try:
            global_step = tf.train.get_global_step(sess.graph)
            print('started training...')
            for epoch in range(flags.epochs):
                for step in range(0, n_train, batch_size):

                    X_batch, y_batch, box_batch = get_data_batch(train_data, n_train, batch_size)
                    #print(X_batch.shape, y_batch.shape, box_batch.shape)
                    _, step_iou, global_step_value, box_loss = sess.run(
                        [train_op, get_iouop, global_step, box_loss_op],
                        feed_dict={X: X_batch,
                                   y: y_batch,
                                   box_true: box_batch,
                                   mode: True})
                    if (step//batch_size) % 100 == 1:
                        print('<====step/epoch: {}/{} iou: {} box_loss: {}'\
                                   .format(step, epoch, step_iou, box_loss))
                        saver.save(sess, "{}/model.ckpt".format(flags.checkpoint_dir))

                if epoch % 40 == 5:
                    total_iou = 0
                    total_box_loss = 0
                    for step in range(0, n_test, batch_size):
                        X_test, y_test, box_test = get_data_batch(test_data, n_test, batch_size)
                        step_iou, box_loss = sess.run(
                            [get_iouop, box_loss_op],
                            feed_dict={X: X_test,
                                       y: y_test,
                                       box_true: box_test,
                                       mode: False})
                        total_box_loss  += box_loss * X_test.shape[0]
                        total_iou += step_iou * X_test.shape[0]
                    print('<==============validation iou: {}  box loss: {}'\
                        .format(total_iou/n_test, total_box_loss/n_test))

        finally:
            saver.save(sess, "{}/model.ckpt".format(flags.checkpoint_dir))


def overlap_box(rand_box, box):
    x, y = False, False
    if box[0] < rand_box[2] and box[2] > rand_box[0]:
        x = True
    if box[3] > rand_box[1] and box[1] < rand_box[3]:
        y = True
    return x and y

 
def overlap(rand_box, avoid_boxes):
    for box in avoid_boxes:
        if overlap_box(rand_box, box):
          return True
    return False

def post_process(y_pred, box):
    threshold = 0.5
    mask_pred = y_pred > threshold
    
    #labeled_heatmap, n_labels = label(mask_pred)
    labeled_heatmap = mask_pred.astype(np.int32)
    n_labels = 1
    if np.sum(labeled_heatmap) == 0:
        n_labels = 0
    bbox = []
    
    for i in range(n_labels):
        mask_i = labeled_heatmap == (i + 1)
        
        nonzero = np.nonzero(mask_i)
        
        nonzero_row = nonzero[0]
        nonzero_col = nonzero[1]
        
        left_top = [min(nonzero_col), min(nonzero_row)]
        right_bot = [max(nonzero_col), max(nonzero_row)]
        
        if not overlap(left_top + right_bot, bbox):        
            bbox.append(list(left_top + right_bot))
    return bbox, mask_pred * 255

    
def draw_boxes(ret, boxes, tags):
    for box, tag in zip(boxes, tags):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) #(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        ret = cv2.rectangle(ret, (x1,y1), (x2,y2), color=color, thickness=3) 
        font_size = .5
        height, width, __= ret.shape
        #print(tag)
        if y1 < 40:
        	y = y2 + 40
        else:
        	y = y1 - 10
        cv2.putText(ret, tag, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness=1, lineType=cv2.LINE_AA) 
    return ret

def visualize(img, boxes, length_list, digits_list):
    if len(boxes) == 0:
        return img

    tags = []
    for length, digits in zip(length_list, digits_list):
        tags.append("{} digits ({})".format(length, digits[:length].decode('ASCII')))
    ret = draw_boxes(img, boxes, tags)
    return ret


def visualize_box(img, boxes):
    if len(boxes) == 0:
        return img

    tags = []
    for box in boxes:
        tags.append("{},{}".format("", ""))

    ret = draw_boxes(img, boxes, tags)
    return ret

def resize_img(img):
    h, w, _ =img.shape
    h, w = h, w
    hh = (h//32) * 32
    ww = (w//32) * 32
    #print(img.shape, hh, ww)
    return cv2.resize(img, (ww, hh))


def test(flags):
    test_data, n_test = get_data(test_data_dir)
    batch_size = flags.batch_size if train_mode == 'val' else 1

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[batch_size, None, None, 1], name="y")
    box_true = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name="box_true")
    mode = tf.placeholder(tf.bool, name="mode")

    pred, box_pred = unet(X, mode, flags)

    get_iouop = get_iou(pred, y)
    box_loss_op = tf.reduce_mean(tf.square(box_pred-box_true) * y) / (tf.reduce_mean(y) + 1e-4)


    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.exists(flags.checkpoint_dir) and tf.train.checkpoint_exists(
                flags.checkpoint_dir):
            latest_check_point = tf.train.latest_checkpoint(flags.checkpoint_dir)
            saver.restore(sess, latest_check_point)

        else:
            try:
                os.rmdir(flags.checkpoint_dir)
            except FileNotFoundError:
                raise '[ERROR] Checkpoint file not found'

        if train_mode == 'val':
            total_iou = 0
            total_box_loss = 0
            for step in range(0, n_test, batch_size):
                X_test, y_test, box_test = get_data_batch(test_data, n_test, batch_size)
                step_iou, step_summary, box_loss = sess.run(
                        [get_iouop, summary_op, box_loss_op],
                        feed_dict={X: X_test,
                                   y: y_test,
                                   box_true: box_test,
                                   mode: False})
                total_box_loss  += box_loss * X_test.shape[0]
                total_iou += step_iou * X_test.shape[0]
            print('<==============validation iou: {}  box loss: {}'\
                    .format(total_iou/n_test, total_box_loss/n_test))
        else:
            img_paths = glob.glob(test_dir  + '/*[g|G]')
            for img_path in img_paths:
                img = cv2.imread(img_path)
                img = resize_img(img)
                y_pred, box = sess.run(
                        [pred, box_pred],
                        feed_dict={X: [img], mode: False})
                box_list, label_map = post_process(y_pred[0], box[0])
                print(y_pred.shape)
                basename = os.path.basename(img_path)
                res = visualize_box(img, box_list)
                cv2.imwrite(output_dir + '/' + basename + '-heat.jpg', y_pred[0]*255)
                cv2.imwrite(output_dir + '/' + basename + '-label.jpg', label_map)
                cv2.imwrite(output_dir + '/' + basename + '-vis.jpg', res)

def get_DDN(sess, flags):
    batch_size = 1
    X = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name="X")
    pred, box_pred = unet(X, False, flags)

    ddn_variables = [v for v in tf.global_variables() \
    	     if v.name.startswith('down') or v.name.startswith('up') or \
    	     v.name.startswith('dilat') or v.name.startswith('y_op')]

    init = tf.variables_initializer(ddn_variables)
    sess.run(init)
    saver = tf.train.Saver(ddn_variables)
    if os.path.exists(flags.DDN_checkpoint_dir):
            latest_check_point = tf.train.latest_checkpoint(flags.DDN_checkpoint_dir)
            saver.restore(sess, latest_check_point)
    else:
            raise '[ERROR] Checkpoint file not found'
    return [X, pred]

def detect(sess, DDN, img):
    X, pred = DDN
    img = resize_img(img)
    y_pred = sess.run(pred,feed_dict={X: [img]})
    box_list, label_map = post_process(y_pred[0], None)
    #print(y_pred.shape)
    return box_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", default=5000, type=int, help="# of epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--reg", type=float, default=0.1, help="coefficient of regularization term")
    parser.add_argument("--checkpoint_dir", default="ckpt", help="Checkpoint directory")
    flags = parser.parse_args()
    l2_alpha = 0. # 1.
    train_data_dir = '../data/DigitBox'
    test_data_dir = '../data/DigitBox'
    test_dir =  '../data/DigitBox'
    output_dir = './tests/'
    train_mode = 'train' # 'test', 'val', 'train'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if train_mode == 'train':
        train(flags)
    else:
        test(flags)

