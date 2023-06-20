from params import *
import argparse
import os
import numpy as np

from model import CycleGAN
from preprocess import *
import soundfile as sf


def conversion(file, conversion_direction='A2B'):

    num_features = 24
    sampling_rate = audio_sampling_rate
    frame_period = 5.0

    model = CycleGAN(num_features = num_features, mode = 'test')

    model_name = "{}.ckpt".format(model_prefix)
    model.load(filepath = os.path.join(model_dir, model_name))

    mcep_normalization_params = np.load(os.path.join(norm_dir, f'{model_prefix}_logf0s_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(norm_dir, f'{model_prefix}_mcep_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav, _ = librosa.load(file, sr = sampling_rate, mono = True)
    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
    print('Shape of F0: {}'.format(f0.shape))
    print('Shape of SP: {}'.format(sp.shape))
    print('Shape of AP: {}'.format(ap.shape))
    print()

    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
    print('Shape of Coded Sp : {}'.format(coded_sp.shape))

    coded_sp_transposed = coded_sp.T
    print('Shape of Coded Sp transpose {}'.format(coded_sp_transposed.shape))


    if conversion_direction == 'A2B':
        f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A, mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
        #f0_converted = f0
        coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A
        print('Shape of Coded Sp Norm: {}'.format(coded_sp_norm.shape))
        coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
        coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
    else:
        f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_B, std_log_src = logf0s_std_B, mean_log_target = logf0s_mean_A, std_log_target = logf0s_std_A)
        #f0_converted = f0
        coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B
        coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
        coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A
   
    print('Shape of coded Sp Converted Norm: {}'.format(coded_sp_converted_norm.shape))
    print('Shape of Coded SP Converted before transpose: {}'.format(coded_sp_converted.shape))
    print()
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)

    print('Shape of F0 Converted: {}'.format(f0_converted.shape))
    print('Shape of SP Converted: {}'.format(decoded_sp_converted.shape))
    print('Shape of Coded SP Converted after transpose: {}'.format(coded_sp_converted.shape))

    # Get the portion of A from B's shape[0] to the last row
    # values = sp[decoded_sp_converted.shape[0]:, :]

    # # Concatenate the portion of A and B
    # transformed_decoded_sp_converted = np.concatenate((decoded_sp_converted, values), axis=0)


    # print('Shape of F0 after truncation: {}'.format(f0.shape))
    # print('Shape of SP after truncation: {}'.format(transformed_decoded_sp_converted.shape))
    # print('Shape of AP  after truncation: {}'.format(ap.shape))
    wav_transformed = world_speech_synthesis(f0 = f0, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)

    visualize_audio(wav,sampling_rate,'Monotone audio')
    visualize_audio(wav_transformed,sampling_rate,'Synthesised audio')
    sf.write(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate)



