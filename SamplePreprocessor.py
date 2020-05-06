import os
import cv2
import string
import numpy as np
from keras import backend as bknd
import math
import random

import tensorflow as tf
from PIL import Image, ImageFilter
from load_data import imageprepare
from pre import preprocess

import tempfile


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (32, 32), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((30 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((30, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((32 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((30 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth, 30), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((32 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    # newImage.save("sample.png")
    tv = list(newImage.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) / 255 for x in tv]
    return tva


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return X, Y, keep_prob


def forward_propagation(X, keep_prob):
    tf.set_random_seed(1)

    # CONV 1
    W1 = tf.get_variable("W1", [4, 4, 1, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1", [64, 1], initializer=tf.zeros_initializer())

    Z1 = tf.nn.conv2d(input=X, filter=W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    # MAX POOL 1
    P1 = tf.nn.max_pool(A1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # CONV 2
    W2 = tf.get_variable("W2", [2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", [128, 1], initializer=tf.zeros_initializer())

    Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    # MAX POOL 2
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULL CONNECT 1
    W3 = tf.get_variable('W3', [512, P2.shape[1:2].num_elements()],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable('b3', [512, 1], initializer=tf.zeros_initializer())
    Z3 = tf.add(tf.matmul(W3, tf.matrix_transpose(P2)), b3)
    A3 = tf.nn.relu(Z3)
    # FULL CONNECT 2
    W4 = tf.get_variable('W4', [256, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable('b4', [256, 1], initializer=tf.zeros_initializer())
    A4_drop = tf.nn.dropout(A3, keep_prob)
    Z4 = tf.add(tf.matmul(W4, A4_drop), b4)
    A4 = tf.nn.relu(Z4)
    # FULL CONNECT 3
    W5 = tf.get_variable('W5', [63, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable('b5', [63, 1], initializer=tf.zeros_initializer())
    A5_drop = tf.nn.dropout(A4, keep_prob)
    Z5 = tf.add(tf.matmul(W5, A5_drop), b5)

    Z5 = tf.matrix_transpose(Z5)

    return Z5


def draw(result):
    a = result[0]
    #print("result",a)

    if a <= 9:
        return a
    # else:
    #     return None

    elif a >= 10 and a <= 35:
        return chr(a + 87)
    else:
        return chr(a + 29)


def remove_line(box_bw, line_thickness):
    edges = cv2.Canny(box_bw, 80, 120)

    # dilate: it will fill holes between line segments
    (r, c) = np.shape(box_bw)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    edges = cv2.dilate(edges, element)
    min = np.minimum(r, c)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 2, None, min * 0.75, 1);

    r_low_lim = r * 0.1
    r_high_lim = r - r_low_lim

    c_low_lim = c * 0.1
    c_high_lim = c - c_low_lim

    if lines != None:
        for line in lines[0]:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            theta_radian2 = np.arctan2(line[2] - line[0],
                                       line[3] - line[1])  # calculating the slope and the result returned in radian!
            theta_deg2 = (180 / math.pi) * theta_radian2  # converting radian into degrees!
            if (theta_deg2 > 85 and theta_deg2 < 95):  # horizontal line
                # if starting of line is below or above 30% of box, remove it
                if (line[1] <= r_low_lim or line[1] >= r_high_lim) and (line[3] <= r_low_lim or line[3] >= r_high_lim):
                    cv2.line(box_bw, pt1, pt2, 255, line_thickness)
            if (theta_deg2 > 175 and theta_deg2 < 185):  # vertical line
                if (line[0] <= c_low_lim or line[0] >= c_high_lim) and (line[2] <= c_low_lim or line[2] >= c_high_lim):
                    cv2.line(box_bw, pt1, pt2, 255, line_thickness)

    return box_bw

# border removal by inpainting
def border_removal(box_bw, top, bottom, right, left):
    box_bw[0:top, :] =  255  # first "top"  number of rows
    box_bw[-bottom:, ] = 255  # last "bottom" number of rows
    box_bw[:, 0:left] = 255  # first "left" number of columns
    box_bw[:, -right:] = 255 # last "right" number of columns
    # last two rows a[-2:,]
    return box_bw

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def chunk_preprocessor(img_chunk):
    height, width = img_chunk.shape[:2]
    (thresh, box_bw) = cv2.threshold(img_chunk, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    top = 5
    bottom = 5
    right = 5
    left = 2
    box_bw_border_free = border_removal(box_bw, top, bottom, right, left)
    #box_bw_border_free = remove_border(box_bw_border_free)
    box_bw_border_free = cv2.medianBlur(box_bw_border_free, 3)
    box_bw_border_free = cv2.GaussianBlur(box_bw_border_free, (1,1),0)
    (thresh, In_bw) = cv2.threshold(box_bw_border_free, 128, 255, cv2.THRESH_BINARY_INV)
    inverted_In_bw = np.invert(In_bw)
    (i, j) = np.nonzero(inverted_In_bw)
    if np.size(i) != 0:  # in case the box contains no BLACK pixel(i.e. the box is empty such as checkbox)
        Out_cropped = box_bw_border_free[np.min(i):np.max(i),
                  np.min(j):np.max(j)]  # row column operation to extract the non

    else:  # no need to do cropping since its an empty box
        Out_cropped = box_bw_border_free

    ker = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    Out_cropped = cv2.erode(Out_cropped, ker)
    Out_cropped = remove_noise_and_smooth(Out_cropped)
    return Out_cropped

def numb(img):
    seed = 3
    tf.reset_default_graph()
    tf.set_random_seed(seed)


    m, n_H0, n_W0, n_C0 = 1, 32, 32, 1
    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, 63)

    Z3 = forward_propagation(X, keep_prob)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    new_saver = tf.train.import_meta_graph('my-model.ckpt.meta')
    img = cv2.imread(img)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = preprocess(frame)
    cv2.imwrite('draw.jpg', frame)
    tva = imageprepare('draw.jpg')
    image = np.array(tva)
    np.save('frame.npy', image)
    image = image.reshape(1, n_H0, n_W0, 1)
    prediction = tf.argmax(Z3, 1)

    with tf.Session() as sess:
        sess.run(init)
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        result = sess.run(prediction, feed_dict={X: image, keep_prob: 0.9})
        res=draw(result)
    return res



