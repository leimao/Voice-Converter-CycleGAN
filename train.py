

import tensorflow as tf
import os
import numpy as np
import argparse
import time
import librosa

#from preprocess import load_wavs, world_decompose, world_encode_spectral_envelop, world_encode_data, sample_train_data, coded_sps_normalization_fit_transoform, transpose_in_list
from preprocess import *
from model import CycleGAN


def train(train_A_dir = './data/vcc2016_training/SF1', train_B_dir = './data/vcc2016_training/TM1', model_dir = './model', model_name = 'female_male.ckpt', random_seed = 0, validation_A_dir = './data/evaluation_all/SF1', validation_B_dir = './data/evaluation_all/TM1', output_dir = './validation_output'):

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
    n_mcep = 24
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5

    print('Preprocessing Data ...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir = train_A_dir, sr = sampling_rate)
    wavs_B = load_wavs(wav_dir = train_B_dir, sr = sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = n_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = n_mcep)

    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)


    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)


    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CycleGAN(num_features = n_mcep)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        if epoch > 100:
            lambda_identity = 0
        if epoch > 200:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], lambda_cycle = lambda_cycle, lambda_identity = lambda_identity, generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate)

            if i % 50 == 0:
                print('Minibatch: %d, Generator Loss : %f, Discriminator Loss : %f' % (i, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = model_name)


        if validation_A_dir is not None:
            for file in os.listdir(validation_A_dir):
                filepath = os.path.join(validation_A_dir, file)
                wav, _ = librosa.load(file_path, sr = sampling_rate, mono = True)
                wav = wav.astype(np.float64)
                f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = n_mcep)
                coded_sp_transposed = np.array([coded_sp.T])
                coded_sp_norm = coded_sps_normalization_transoform(coded_sps = coded_sp_transposed, coded_sps_mean = coded_sps_A_mean, coded_sps_std = coded_sps_A_std)
                coded_sp_converted_norm = model.test(inputs = coded_sp_norm, direction = 'A2B')[0]
                coded_sp_converted = coded_sps_normalization_inverse_transoform(normalized_coded_sps = coded_sp_converted_norm, coded_sps_mean = coded_sps_A_mean, coded_sps_std = coded_sps_A_std)
                coded_sp_converted = coded_sp_converted.T
                decoded_sp_converted = world_decode_data(coded_sps = coded_sp_converted, fs = sampling_rate)
                wav_transformed = world_speech_synthesis(f0 = f0, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


        if validation_B_dir is not None:
            for file in os.listdir(validation_B_dir):
                filepath = os.path.join(validation_B_dir, file)
                wav, _ = librosa.load(file_path, sr = sampling_rate, mono = True)
                wav = wav.astype(np.float64)
                f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = n_mcep)
                coded_sp_transposed = np.array([coded_sp.T])
                coded_sp_norm = coded_sps_normalization_transoform(coded_sps = coded_sp_transposed, coded_sps_mean = coded_sps_B_mean, coded_sps_std = coded_sps_B_std)
                coded_sp_converted_norm = model.test(inputs = coded_sp_norm, direction = 'B2A')[0]
                coded_sp_converted = coded_sps_normalization_inverse_transoform(normalized_coded_sps = coded_sp_converted_norm, coded_sps_mean = coded_sps_B_mean, coded_sps_std = coded_sps_B_std)
                coded_sp_converted = coded_sp_converted.T
                decoded_sp_converted = world_decode_data(coded_sps = coded_sp_converted, fs = sampling_rate)
                wav_transformed = world_speech_synthesis(f0 = f0, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                librosa.output.write_wav(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


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