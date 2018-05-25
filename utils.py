
import tensorflow as tf
import os
import random
import numpy as np
import cv2


def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

def image_scaling(imgs):

    imgs_scaled = imgs / 127.5 - 1

    return imgs_scaled

def image_scaling_inverse(imgs):

    imgs_rescaled = (imgs + 1) * 127.5

    return imgs_rescaled


def read_img_modified(img_filepath, load_size = 286, output_size = 256):
    
    # We first enlarge the original image and downsample the enlarged image to add more data to the train set

    assert load_size >= output_size

    img = cv2.imread(img_filepath)

    img_enlarged = cv2.resize(img, (load_size, load_size))

    h_start = np.random.randint(load_size - output_size + 1)
    h_end = h_start + output_size
    w_start = np.random.randint(load_size - output_size + 1)
    w_end = w_start + output_size

    img_output = img_enlarged[h_start:h_end, w_start:w_end]

    # Flip image in the left/right direction
    if np.random.random() >  0.5:
        img_output = np.fliplr(img_output)

    # Image scaling
    img_output = image_scaling(imgs = img_output)

    return img_output


def load_train_data(img_A_dir, img_B_dir, load_size = 286, output_size = 256):

    img_A_filepaths = [os.path.join(img_A_dir, file) for file in os.listdir(img_A_dir) if os.path.isfile(os.path.join(img_A_dir, file))]
    img_B_filepaths = [os.path.join(img_B_dir, file) for file in os.listdir(img_B_dir) if os.path.isfile(os.path.join(img_B_dir, file))]

    num_samples = min(len(img_A_filepaths), len(img_B_filepaths))

    random.shuffle(img_A_filepaths)
    random.shuffle(img_B_filepaths)

    img_A_filepaths_sampled = img_A_filepaths[:num_samples]
    img_B_filepaths_sampled = img_B_filepaths[:num_samples]

    img_A_dataset = [read_img_modified(img_filepath = filepath, load_size = load_size, output_size = output_size) for filepath in img_A_filepaths_sampled]
    img_B_dataset = [read_img_modified(img_filepath = filepath, load_size = load_size, output_size = output_size) for filepath in img_B_filepaths_sampled]

    img_A_dataset = np.array(img_A_dataset)
    img_B_dataset = np.array(img_B_dataset)

    return img_A_dataset, img_B_dataset



def load_data(img_dir, load_size = 256):

    img_filepaths = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
    img_dataset = [cv2.resize(cv2.imread(filepath), (load_size, load_size)) for filepath in img_filepaths]
    img_dataset = np.array(img_dataset)

    return img_dataset


def img_subsampling(img, load_size, output_size):

    img_enlarged = cv2.resize(img, (load_size, load_size))

    h_start = np.random.randint(load_size - output_size + 1)
    h_end = h_start + output_size
    w_start = np.random.randint(load_size - output_size + 1)
    w_end = w_start + output_size

    img_output = img_enlarged[h_start:h_end, w_start:w_end]

    # Flip image in the left/right direction
    if np.random.random() >  0.5:
        img_output = np.fliplr(img_output)

    return img_output


def sample_train_data(img_A_dataset, img_B_dataset, load_size = 286, output_size = 256):

    num_samples = min(len(img_A_dataset), len(img_B_dataset))
    train_data_A_idx = np.arange(len(img_A_dataset))
    train_data_B_idx = np.arange(len(img_B_dataset))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)

    img_A_subset = img_A_dataset[train_data_A_idx[:num_samples]]
    img_B_subset = img_B_dataset[train_data_B_idx[:num_samples]]

    train_data_A = list()
    train_data_B = list()

    for img in img_A_subset:
        img_output = img_subsampling(img = img, load_size = load_size, output_size = output_size)
        img_output = image_scaling(imgs = img_output)
        train_data_A.append(img_output)

    for img in img_B_subset:
        img_output = img_subsampling(img = img, load_size = load_size, output_size = output_size)
        img_output = image_scaling(imgs = img_output)
        train_data_B.append(img_output)

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B









