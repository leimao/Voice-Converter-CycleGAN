import argparse
import os
import numpy as np

from model import CycleGAN
from preprocess import *

def conversion(model_dir, model_name, data_dir, conversion_direction, output_dir):

    num_features = 24
    sampling_rate = 16000
    frame_period = 5.0

    model = CycleGAN(input_size = input_size, num_filters = num_filters, mode = 'test')

    model.load(filepath = os.path.join(model_dir, model_name))

    normalization_params = np.load(os.path.join(model_dir, 'normalization.npz'))
    mean_A = normalization_params['mean_A']
    std_A = normalization_params['std_A']
    mean_B = normalization_params['mean_B']
    std_B = normalization_params['std_B']

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
            coded_sp_norm = (coded_sp_transposed - mean_A) / std_A
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * std_B + mean_B
        else:
            coded_sp_norm = (coded_sp_transposed - mean_B) / std_B
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * std_A + mean_A

        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
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
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted images.', default = output_dir_default)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir

    conversion(model_dir = model_dir, model_name = model_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir)


