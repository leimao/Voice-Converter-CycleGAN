

import tensorflow as tf
import os
import numpy as np
import argparse
import time

from preprocess import load_wavs, wavs_to_mfccs, sample_train_data, mfccs_normalization
from model import CycleGAN


def train(voice_A_dir = './data/vcc2016_training/SF1', voice_B_dir = './data/vcc2016_training/TM1', model_dir = './model', model_name = 'female_male.ckpt', random_seed = 0):

    np.random.seed(random_seed)

    num_epochs = 2000
    mini_batch_size = 1 # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    discriminator_learning_rate = 0.0001
    num_features = 24
    sampling_rate = 16000
    n_fft = 256
    hop_length = n_fft // 4
    n_mels = 128
    n_mfcc = 24
    n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5

    wavs_A = load_wavs(wav_dir = voice_A_dir, sr = sampling_rate)
    wavs_B = load_wavs(wav_dir = voice_B_dir, sr = sampling_rate)

    mfccs_A = wavs_to_mfccs(wavs = wavs_A, sr = sampling_rate, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
    mfccs_B = wavs_to_mfccs(wavs = wavs_B, sr = sampling_rate, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)

    mfccs_A_norm, mfccs_A_mean, mfccs_A_std = mfccs_normalization(mfccs = mfccs_A)
    mfccs_B_norm, mfccs_B_mean, mfccs_B_std = mfccs_normalization(mfccs = mfccs_B)

    model = CycleGAN(num_features = 24)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        if epoch > 100:
            lambda_identity = 0
        if epoch > 200:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = mfccs_A_norm, dataset_B = mfccs_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], lambda_cycle = lambda_cycle, lambda_identity = lambda_identity, generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate)

            if i % 50 == 0:
                print('Minibatch: %d, Generator Loss : %f, Discriminator Loss : %f' % (i, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))


if __name__ == '__main__':

    train()


'''

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
'''