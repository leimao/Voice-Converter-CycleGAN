
import argparse
import cv2
import os
import numpy as np

from model import CycleGAN
from utils import load_data, sample_train_data, image_scaling, image_scaling_inverse

def conversion(model_filepath, img_dir, conversion_direction, output_dir):

    input_size = [256, 256, 3]
    num_filters = 8

    model = CycleGAN(input_size = input_size, num_filters = num_filters, mode = 'test')

    model.load(filepath = model_filepath)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(img_dir):
        filepath = os.path.join(img_dir, file)
        img = cv2.imread(filepath)
        img_height, img_width, img_channel = img.shape
        img = cv2.resize(img, (input_size[1], input_size[0]))
        img = image_scaling(imgs = img)
        img_converted = model.test(inputs = np.array([img]), direction = conversion_direction)[0]
        img_converted = image_scaling_inverse(imgs = img_converted)
        img_converted = cv2.resize(img_converted, (img_width, img_height))
        cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), img_converted)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert images using pre-trained CycleGAN model.')

    model_filepath_default = './model/horse_zebra/horse_zebra.ckpt'
    img_dir_default = './data/horse2zebra/testA'
    conversion_direction_default = 'A2B'
    output_dir_default = './converted_images'

    parser.add_argument('--model_filepath', type = str, help = 'File path for the pre-trained model.', default = model_filepath_default)
    parser.add_argument('--img_dir', type = str, help = 'Directory for the images for conversion.', default = img_dir_default)
    parser.add_argument('--conversion_direction', type = str, help = 'Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.', default = conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted images.', default = output_dir_default)

    argv = parser.parse_args()

    model_filepath = argv.model_filepath
    img_dir = argv.img_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir

    conversion(model_filepath = model_filepath, img_dir = img_dir, conversion_direction = conversion_direction, output_dir = output_dir)