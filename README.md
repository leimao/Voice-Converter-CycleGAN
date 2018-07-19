# Voice Converter CycleGAN

Lei Mao

University of Chicago


## Introduction

Cycle-consistent adversarial networks (CycleGAN) has been widely used for image conversions. It turns out that it could also be used for voice conversion. This is an implementation of CycleGAN on human speech conversions. The neural network utilized 1D gated convolution neural network (Gated CNN) for generator, and 2D Gated CNN for discriminator. The model takes Mel-cepstral coefficients ([MCEPs](https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification/wiki/Feature-Extraction-for-Speech-Spoofing)) (for spectral envelop) as input for voice conversions.

<p align="center">
    <img src = "./figures/teaser_high_res.png" width="100%">
</p>


## Dependencies

* Python 3.5
* Numpy 1.14
* TensorFlow 1.8
* ProgressBar2 3.37.1
* LibROSA 0.6
* FFmpeg 4.0
* [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)


## Files

```
.
├── convert.py
├── demo
├── download.py
├── figures
├── LICENSE.md
├── model.py
├── module.py
├── preprocess.py
├── README.md
├── train_log
├── train.py
└── utils.py
```

## Usage

### Download Dataset

Download and unzip [VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211) dataset to designated directories.

```bash
$ python download.py --help
usage: download.py [-h] [--download_dir DOWNLOAD_DIR] [--data_dir DATA_DIR]
                   [--datasets DATASETS]

Download CycleGAN voice conversion datasets.

optional arguments:
  -h, --help            show this help message and exit
  --download_dir DOWNLOAD_DIR
                        Download directory for zipped data
  --data_dir DATA_DIR   Data directory for unzipped data
  --datasets DATASETS   Datasets available: vcc2016
```

For example, to download the datasets to ``download`` directory and extract to ``data`` directory:

```bash
$ python download.py --download_dir ./download --data_dir ./data --datasets vcc2016
```

### Train Model

To have a good conversion capability, the training would take at least 1000 epochs, which could take very long time even using a NVIDIA GTX TITAN X graphic card. 

```bash
$ python train.py --help
usage: train.py [-h] [--train_A_dir TRAIN_A_DIR] [--train_B_dir TRAIN_B_DIR]
                [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                [--random_seed RANDOM_SEED]
                [--validation_A_dir VALIDATION_A_DIR]
                [--validation_B_dir VALIDATION_B_DIR]
                [--output_dir OUTPUT_DIR]
                [--tensorboard_log_dir TENSORBOARD_LOG_DIR]

Train CycleGAN model for datasets.

optional arguments:
  -h, --help            show this help message and exit
  --train_A_dir TRAIN_A_DIR
                        Directory for A.
  --train_B_dir TRAIN_B_DIR
                        Directory for B.
  --model_dir MODEL_DIR
                        Directory for saving models.
  --model_name MODEL_NAME
                        File name for saving model.
  --random_seed RANDOM_SEED
                        Random seed for model training.
  --validation_A_dir VALIDATION_A_DIR
                        Convert validation A after each training epoch. If set
                        none, no conversion would be done during the training.
  --validation_B_dir VALIDATION_B_DIR
                        Convert validation B after each training epoch. If set
                        none, no conversion would be done during the training.
  --output_dir OUTPUT_DIR
                        Output directory for converted validation voices.
  --tensorboard_log_dir TENSORBOARD_LOG_DIR
                        TensorBoard log directory.
```

For example, to train CycleGAN model for voice conversion between ``SF1`` and ``TM1``:

```bash
$ python train.py --train_A_dir ./data/vcc2016_training/SF1 --train_B_dir ./data/vcc2016_training/TM1 --model_dir ./model/sf1_tm1 --model_name sf1_tm1.ckpt --random_seed 0 --validation_A_dir ./data/evaluation_all/SF1 --validation_B_dir ./data/evaluation_all/TM1 --output_dir ./validation_output --tensorboard_log_dir ./log
```


<p align="center">
    <img src = "./train_log/discriminator_discriminator.png" width="90%">
</p>

<p align="center">
    <img src = "./train_log/cycle_identity.png" width="90%">
</p>

With ``validation_A_dir``, ``validation_B_dir``, and ``output_dir`` set, we could monitor the conversion of validation voices after each epoch using our bare ear. 


### Voice Conversion

Convert voices using pre-trained models.

```bash
$ python convert.py --help
usage: convert.py [-h] [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                  [--data_dir DATA_DIR]
                  [--conversion_direction CONVERSION_DIRECTION]
                  [--output_dir OUTPUT_DIR]

Convert voices using pre-trained CycleGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory for the pre-trained model.
  --model_name MODEL_NAME
                        Filename for the pre-trained model.
  --data_dir DATA_DIR   Directory for the voices for conversion.
  --conversion_direction CONVERSION_DIRECTION
                        Conversion direction for CycleGAN. A2B or B2A. The
                        first object in the model file name is A, and the
                        second object in the model file name is B.
  --output_dir OUTPUT_DIR
                        Directory for the converted voices.
```

To convert voice, put wav-formed speeches into ``data_dir`` and run the following commands in the terminal, the converted speeches would be saved in the ``output_dir``:

```bash
$ python convert.py --model_dir ./model/sf1_tm1 --model_name sf1_tm1.ckpt --data_dir ./data/evaluation_all/SF1 --conversion_direction A2B --output_dir ./converted_voices
```
The convention for ``conversion_direction`` is that the first object in the model filename is A, and the second object in the model filename is B. In this case, ``SF1 = A`` and ``TM1 = B``.

## Demo

### VCC2016 SF1 and TF2 Conversion

In the ``demo`` directory, there are voice conversions between the validation data of ``SF1`` and ``TF2`` using the pre-trained model.

``200001_SF1.wav`` and ``200001_TF2.wav`` are real voices for the same speech from ``SF1`` and ``TF2``, respectively.

``200001_SF1toTF2.wav`` and ``200001_TF2.wav`` are the converted voice using the pre-trained model.

``200001_SF1toTF2_author.wav`` is the converted voice from the [NTT](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/) website for comparison with our model performance.

The conversion performance is extremely good and the converted speech sounds real to me.

Download the pre-trained SF1-TF2 conversion model and conversion of all the validation samples from [Google Drive](https://drive.google.com/open?id=1SwiK9X3crXU4_-aM_-Sff1T82d6-1SEg).


## Reference

* Takuhiro Kaneko, Hirokazu Kameoka. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. 2017. (Voice Conversion CycleGAN)
* Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. 2016. (Pixel Shuffler)
* Yann Dauphin, Angela Fan, Michael Auli, David Grangier. Language Modeling with Gated Convolutional Networks. 2017. (Gated CNN)
* Takuhiro Kaneko, Hirokazu Kameoka, Kaoru Hiramatsu, Kunio Kashino. Sequence-to-Sequence Voice Conversion with Similarity Metric Learned Using Generative Adversarial Networks. 2017. (1D Gated CNN)
* Kun Liu, Jianping Zhang, Yonghong Yan. High Quality Voice Conversion through Phoneme-based Linear Mapping Functions with STRAIGHT for Mandarin. 2007. (Foundamental Frequnecy Transformation)
* [PyWorld and SPTK Comparison](http://nbviewer.jupyter.org/gist/r9y9/ca05349097b2a3926ec77a02e62c6632)
* [Gated CNN TensorFlow](https://github.com/anantzoid/Language-Modeling-GatedCNN)

## To-Do List

- [ ] Parallelize data preprocessing
- [ ] Evaluation metrics
- [x] Hyper parameter tuning
- [ ] Train more conversion models
- [x] Argparse