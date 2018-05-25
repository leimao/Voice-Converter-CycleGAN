

import tensorflow as tf
import cv2
import os
import numpy as np
import argparse
import time

from utils import load_data, sample_train_data, image_scaling, image_scaling_inverse
from model import CycleGAN

def train(img_A_dir, img_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir):

    np.random.seed(random_seed)

    num_epochs = 200
    mini_batch_size = 1 # mini_batch_size = 1 is better
    learning_rate = 0.0002
    input_size = [256, 256, 3]
    num_filters = 8 

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    model = CycleGAN(input_size = input_size, num_filters = num_filters, mode = 'train')

    dataset_A_raw = load_data(img_dir = img_A_dir, load_size = 256)
    dataset_B_raw = load_data(img_dir = img_B_dir, load_size = 256)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A_raw, dataset_B_raw, load_size = 286, output_size = 256)

        n_samples = dataset_A.shape[0]
        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], learning_rate = learning_rate)

            if i % 50 == 0:
                print('Minibatch: %d, Generator Loss : %f, Discriminator Loss : %f' % (i, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = model_name)

        if validation_A_dir is not None:
            for file in os.listdir(validation_A_dir):
                filepath = os.path.join(validation_A_dir, file)
                img = cv2.imread(filepath)
                img_height, img_width, img_channel = img.shape
                img = cv2.resize(img, (input_size[1], input_size[0]))
                img = image_scaling(imgs = img)
                img_converted = model.test(inputs = np.array([img]), direction = 'A2B')[0]
                img_converted = image_scaling_inverse(imgs = img_converted)
                img_converted = cv2.resize(img_converted, (img_width, img_height))
                cv2.imwrite(os.path.join(validation_A_output_dir, os.path.basename(file)), img_converted)

        if validation_B_dir is not None:
            for file in os.listdir(validation_B_dir):
                filepath = os.path.join(validation_B_dir, file)
                img = cv2.imread(filepath)
                img_height, img_width, img_channel = img.shape
                img = cv2.resize(img, (input_size[1], input_size[0]))
                img = image_scaling(imgs = img)
                img_converted = model.test(inputs = np.array([img]), direction = 'B2A')[0]
                img_converted = image_scaling_inverse(imgs = img_converted)
                img_converted = cv2.resize(img_converted, (img_width, img_height))
                cv2.imwrite(os.path.join(validation_B_output_dir, os.path.basename(file)), img_converted)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    img_A_dir_default = './data/horse2zebra/trainA'
    img_B_dir_default = './data/horse2zebra/trainB'
    model_dir_default = './model/horse_zebra'
    model_name_default = 'horse_zebra.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/horse2zebra/testA'
    validation_B_dir_default = './data/horse2zebra/testB'
    output_dir_default = './validation_output'

    parser.add_argument('--img_A_dir', type = str, help = 'Directory for A images.', default = img_A_dir_default)
    parser.add_argument('--img_B_dir', type = str, help = 'Directory for B images.', default = img_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A images after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B images after each training epoch. If set none, no conversion would be done during the training.', default = validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation images.', default = output_dir_default)

    argv = parser.parse_args()

    img_A_dir = argv.img_A_dir
    img_B_dir = argv.img_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir

    train(img_A_dir = img_A_dir, img_B_dir = img_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir)