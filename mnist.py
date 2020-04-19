#!/usr/bin/env python3

# https://github.com/nst/MNIST

import struct
import sys
import math
#import matplotlib.pyplot as plt

TRAIN_IMAGES_PATH = "train-images-idx3-ubyte"
TRAIN_LABELS_PATH = "train-labels-idx1-ubyte"

TEST_IMAGES_PATH = "t10k-images-idx3-ubyte"
TEST_LABELS_PATH = "t10k-labels-idx1-ubyte"

ROWS = 28
COLS = 28

def read_fmt(f, fmt):
    s = struct.Struct(fmt)
    data = f.read(s.size)
    if len(data) == 0:
        return None
    return s.unpack(data)

def softmax(l):
    maximum = max(l)
    l2 = [x / maximum for x in l]
    s = sum([math.exp(x) for x in l2])
    return [math.exp(x)/s for x in l2]

def distance(img_1, img_2):
    return sum([(a-b)**2 for a, b in zip(img_1, img_2)])

def predict(avg_images, img_test):

    img_test_normalized = softmax(img_test)
    
    distances = [0.0] * 10
    
    for i, img_train in enumerate(avg_images):
        distances[i] = distance(img_test_normalized, img_train)
    
    distances = softmax(distances)
    return distances.index(min(distances)), distances

def gen_data(labels_path, images_path):

    with open(images_path, "rb") as f1:
        with open(labels_path, "rb") as f2:
    
            # read images headers
    
            magic_number = read_fmt(f1, ">I")[0]
            
            if magic_number != 2051:
                print("-- wrong magic number", magic_number)
                sys.exit(1)
            
            img_nb_items = read_fmt(f1, ">I")[0]
            rows = read_fmt(f1, ">I")[0]
            cols = read_fmt(f1, ">I")[0]
            
            assert(rows == ROWS)
            assert(cols == COLS)
            
            img_size = rows * cols
            
            img_fmt = ">%dB" % img_size
    
            # read labels headers
        
            magic_number = read_fmt(f2, ">I")[0]
            
            if magic_number != 2049:
                print("-- wrong magic number", magic_number)
                sys.exit(1)
            
            lbl_nb_items = read_fmt(f2, ">I")[0]
            
            assert(img_nb_items == lbl_nb_items)
    
            lbl_fmt = ">B"
            
            # read images and labels
            
            i = 0
            while True:
                img = read_fmt(f1, img_fmt)
                if not img:
                    break
                
                lbl = read_fmt(f2, lbl_fmt)[0]

                yield i, lbl, img
                i += 1
            
            yield None, None, None

def read_training_data():

    avg_images = []
    for i in range(10):
        avg_images.append([0.0]*COLS*ROWS)

    for i, lbl, img in gen_data(TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH):
        if not img:
            break
    
        avg_images[lbl] = [a + b for a, b in zip(avg_images[lbl], img)]
    
        if i % 5000 == 0:
            print(".", end = ''); sys.stdout.flush()
    
    print("")

    # normalization
    for i, img in enumerate(avg_images):
        avg_images[i] = softmax(avg_images[i])
    
    return avg_images

"""
def draw_average_images(path):

    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    for i, img in enumerate(avg_images):
        lines = [img[x:x+COLS] for x in range(0, len(img), ROWS)]
        a, b = divmod(i, 4)
        ax[a, b].imshow(lines)
    plt.savefig(path)
"""

def read_test_images(avg_images):
    
    good, bad = 0, 0
    
    for index, lbl, img in gen_data(TEST_LABELS_PATH, TEST_IMAGES_PATH):
        if not img:
            break
        
        predicted_label, distances = predict(avg_images, img)
        
        #print(index, "--", lbl, "->", predicted_label)
        #for i, d in enumerate(distances):
        #    print("    ", i, d)
        
        if predicted_label == lbl:
            good += 1
            #save_img(img, "good/%d_%d_%d.png" % (lbl, predicted_label, index), ROWS, COLS)
        else:
            bad += 1
            #save_img(img, "bad/%d_%d_%d.png" % (lbl, predicted_label, index), ROWS, COLS)
    
        if index > 0 and index % 1000 == 0:
            print("    index: %d, accuracy: %f" % (index, good / (good + bad)))

    return good / (good + bad)

if __name__ == '__main__':
    
    print("-- read_training_data()")
    avg_images = read_training_data()

    #print("-- draw_average_images()")
    #draw_average_images("average_digits.png")

    print("-- read_test_images()")
    accuracy = read_test_images(avg_images)
    
    print("-- final accuracy: %f" % accuracy)
