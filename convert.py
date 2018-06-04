import argparse
import os
import numpy as np

from model import CycleGAN
from preprocess import *

def conversion(model_dir, model_name, data_dir, conversion_direction, output_dir):

    num_features = 24
    sampling_rate = 16000
    frame_period = 5.0

    model = CycleGAN(num_features = num_features, mode = 'test')

    model.load(filepath = os.path.join(model_dir, model_name))

    mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):

        filepath = os.path.join(data_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        coded_sp_transposed = coded_sp.T

        if conversion_direction == 'A2B':
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A, mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            #f0_converted = f0
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
        else:
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_B, std_log_src = logf0s_std_B, mean_log_target = logf0s_mean_A, std_log_target = logf0s_std_A)
            #f0_converted = f0
            coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A

        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        librosa.output.write_wav(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')

    model_dir_default = './model/sf1_tm1'
    model_name_default = 'sf1_tm1.ckpt'
    data_dir_default = './data/evaluation_all/SF1'
    conversion_direction_default = 'A2B'
    output_dir_default = './converted_voices'

    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default = model_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
    parser.add_argument('--conversion_direction', type = str, help = 'Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.', default = conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir

    conversion(model_dir = model_dir, model_name = model_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir)


