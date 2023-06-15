from params import *
import os
import numpy as np
import argparse
import time
import librosa
import csv

from preprocess import *
from model import CycleGAN


def prepare_data(train_A_dir, train_B_dir):
    sampling_rate = audio_sampling_rate
    num_mcep = 24
    frame_period = 5.0

    print('Preprocessing Data...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    print('Train data A: {} loaded'.format(len(wavs_A)))
    print('Train data B: {} loaded'.format(len(wavs_B)))
    print()

    f0s_A, coded_sps_A = world_encode_data(
        wavs=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, coded_sps_B = world_encode_data(
        wavs=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    print("Input data fixed.")
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, f'{model_prefix}_logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A, std_A=log_f0s_std_A, mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, f'{model_prefix}_mcep_normalization.npz'),
             mean_A=coded_sps_A_mean, std_A=coded_sps_A_std, mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    return coded_sps_A_norm, coded_sps_B_norm


def train(coded_sps_A_norm, coded_sps_B_norm, random_seed):

    np.random.seed(random_seed)

    num_epochs = num_of_epochs
    mini_batch_size = 1  # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = audio_sampling_rate
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5
    generator_losses = []
    discriminator_losses = []

    model = CycleGAN(num_features=num_mcep)

    for epoch in range(num_epochs):
        print('Epoch: %d/%d' % (epoch, num_epochs))
        '''
        if epoch > 60:
            lambda_identity = 0
        if epoch > 1250:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)
        '''

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(
            dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > 200000:
                generator_learning_rate = max(
                    0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(
                    0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A=dataset_A[start:end], input_B=dataset_B[start:end], lambda_cycle=lambda_cycle,
                                                             lambda_identity=lambda_identity, generator_learning_rate=generator_learning_rate, discriminator_learning_rate=discriminator_learning_rate)
            generator_losses.append(generator_loss)
            discriminator_losses.append(discriminator_loss)

            if i % 50 == 0:
                # print('Iteration: %d, Generator Loss : %f, Discriminator Loss : %f' % (num_iterations, generator_loss, discriminator_loss))
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(
                    num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

            # save losses to a csv file
            losses_csv_path = os.path.join(train_logs__dir, 'losses.csv')

            # Check if the CSV file already exists
            file_exists = os.path.exists(losses_csv_path)

            # Open the CSV file in append mode
            with open(losses_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)

                # Write the title row only if the file doesn't exist
                if not file_exists:
                    writer.writerow(
                        ["num_iterations", "generator_loss", "discriminator_loss"])

                # Write the data row
                writer.writerow(
                    [num_iterations, generator_loss, discriminator_loss])

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch //
              3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

    model_name = "{}.ckpt".format(model_prefix)
    model.save(directory=model_dir, filename=model_name)
    visualize_loss(generator_losses, discriminator_losses,
                   "Generator Losses", "Discriminator Losses", 'Losses')
