
import os
import random
import progressbar
from urllib.request import urlretrieve
import zipfile
import argparse

pbar = None

def progress_bar(block_num, block_size, total_size):
    global pbar
    if pbar is None:

        # pbar = progressbar.ProgressBar(maxval = total_size)
        # Customized progress bar
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(marker = '>', left = '[', right = ']'), ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()] 
        pbar = progressbar.ProgressBar(widgets = widgets, maxval = total_size)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def maybe_download(filename, url, destination_dir, expected_bytes = None, force = False):

    filepath = os.path.join(destination_dir, filename)

    if force or not os.path.exists(filepath):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        print('Attempting to download: ' + filename)
        filepath, _ = urlretrieve(url, filepath, reporthook = progress_bar)
        print('Download complete!')

    statinfo = os.stat(filepath)

    if expected_bytes != None:
        if statinfo.st_size == expected_bytes:
            print('Found and verified: ' + filename)
        else:
            raise Exception('Failed to verify: ' + filename + '. Can you get to it with a browser?')
    else:
        print('Found: ' + filename)
        print('The size of the file: ' + str(statinfo.st_size))

    return filepath


def maybe_unzip(zip_filepath, destination_dir, force = False):

    print('Extracting zip file: ' + os.path.split(zip_filepath)[-1])
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(destination_dir)
    print("Extraction complete!")


def download_vcc2016(download_dir = './download', data_dir = './data'):

    url_prefix = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/'
    data_files = ['vcc2016_training.zip', 'evaluation_all.zip']

    for data_file in data_files:
        url = url_prefix + data_file
        dataset_filepath = maybe_download(filename = data_file, url = url, destination_dir = download_dir, force = False)
        destination_dir = data_dir
        maybe_unzip(zip_filepath = dataset_filepath, destination_dir = destination_dir, force = False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Download CycleGAN voice conversion datasets.')

    download_dir_default = './download'
    data_dir_default = './data'
    datasets_default = 'vcc2016'

    parser.add_argument('--download_dir', type = str, help = 'Download directory for zipped data', default = download_dir_default)
    parser.add_argument('--data_dir', type = str, help = 'Data directory for unzipped data', default = data_dir_default)
    parser.add_argument('--datasets', type = str, help = 'Datasets available: vcc2016', default = datasets_default)

    argv = parser.parse_args()

    download_dir = argv.download_dir
    data_dir = argv.data_dir
    datasets = argv.datasets

    if datasets == 'vcc2016' or datasets == 'VCC2016':
        download_vcc2016(download_dir = download_dir, data_dir = data_dir)
    else:
        print('Dataset not available.')

