import time
import os
import glob
import cv2
import numpy as np
import argparse
from DCN.main import get_DCN, recognize, expand_box
from DDN.main import get_DDN, detect, visualize, visualize_box
import tensorflow as tf
import random

def resize_img(img):
    h, w, _ =img.shape
    h, w = h, w
    scale = 1024. / max(h,w)
    scale = min(1., scale)
    h, w = int(h*scale), int(w*scale)
    hh = (h//32) * 32
    ww = (w//32) * 32
    #print(img.shape, hh, ww)
    return cv2.resize(img, (ww, hh))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DCN_checkpoint_dir", default="./DCN/ckpt", help="Checkpoint directory")
    parser.add_argument("--DDN_checkpoint_dir", default="./DDN/ckpt/", help="# of epochs")
    parser.add_argument("--reg", type=float, default=0.1, help="coefficient of regularization term")
    parser.add_argument("--test_dir", default="./data/imgs", help="Checkpoint directory")
    parser.add_argument("--output_img_dir", default="./graded_imgs", help="Checkpoint directory")
    parser.add_argument("--output_video_dir", default="./graded_imgs/video", help="Checkpoint directory")
    parser.add_argument("--video_path", default="./data/1.avi", help="Checkpoint directory")

    flags = parser.parse_args()
    if not os.path.exists(flags.output_img_dir):
        os.makedirs(flags.output_img_dir)
    if not os.path.exists(flags.output_video_dir):
        os.makedirs(flags.output_video_dir)
    with tf.Session() as sess:
        DCN = get_DCN(sess, flags)
        DDN = get_DDN(sess, flags)
        img_paths = glob.glob(flags.test_dir + '/*[g|G]')
        for img_path in img_paths:
            print('processing img ' + img_path + '...')
            img = cv2.imread(img_path)
            img = resize_img(img)
            #height, width, __= img.shape
            #angle = random.randint(-10, 10)
            #rotation_mat = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
            #img = cv2.warpAffine(img, rotation_mat, (width, height))
            basename = os.path.basename(img_path)
            boxes = detect(sess, DDN, img)
            #res = visualize_box(img, boxes)
            length, digits = recognize(sess, DCN, img, boxes)
            boxes = [expand_box(b) for b in boxes]
            res = visualize(img, boxes, length, digits)
            cv2.imwrite(flags.output_img_dir + '/' + basename[:-4] + '.png', res)
        if flags.video_path is not None:
            print('processing video ' + flags.video_path + '...')
            cap = cv2.VideoCapture(flags.video_path)
            basename = os.path.basename(flags.video_path)
            ret, frame = cap.read()
            if frame is None:
                raise '[ERROR] video file ' + flags.video_path + " can not be read..."
            else:
                height, width, __ = frame.shape
                frame = cv2.resize(frame, (width//2, height//2))
                frame = resize_img(frame)
                height, width, __ = frame.shape

            video_format = cv2.VideoWriter_fourcc('M','J','P','G')
            cap_out = cv2.VideoWriter(flags.output_video_dir + '/' + basename, video_format, 10, (width, height))
            idx = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if frame is None:
                      break
                frame = cv2.resize(frame, (width, height))
                boxes = detect(sess, DDN, frame)
                length, digits = recognize(sess, DCN, frame, boxes)
                boxes = [expand_box(b) for b in boxes]
                res = visualize(frame, boxes, length, digits)
                cap_out.write(res)
                idx += 1
            cap.release()
            cap_out.release()
