# Voice Converter CycleGAN

## Introduction

Cycle-consistent adversarial networks (CycleGAN) has been widely used for image conversions. It turns out that it could also be used for voice conversion. This is an implementation of CycleGAN on human speech conversions. The neural network utilized 1D gated convolution neural network (Gated CNN) for generator, and 2D Gated CNN for discriminator. The model takes Mel-cepstral coefficients ([MCEPs](https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification/wiki/Feature-Extraction-for-Speech-Spoofing)) (for spectral envelop) as input for voice conversions.

## Dependencies

* Python 3.5
* Numpy 1.14
* TensorFlow 1.8
* ProgressBar2 3.37.1
* LibROSA 0.6
* FFmpeg 4.0
* [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)


## Reference

Takuhiro Kaneko, Hirokazu Kameoka. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. 2017.



## To-Do List

- [ ] Parallelize data preprocessing
- [ ] Evaluation metrics
- [ ] Hyper parameter tuning
- [ ] Argparse