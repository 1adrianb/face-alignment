# Face Recognition

Detect facial landmarks from Python using the world's most accurate face alignment network, capable of detection points in both 2D and 3D coordinates.

Build using [FAN](https://www.adrianbulat.com)'s state-of-the-art deep learning based face alignment method. For detecting faces the library makes use of [dlib](http://dlib.net/) library.

<p align="center"><img src="docs/images/face-alignment-adrian.gif" /></p>

Note: For the lua version check [here](https://github.com/1adrianb/2D-and-3D-face-alignment). For numerical evaluations it is recommended to use the lua version which uses indentical models with the ones evaluated in the paper.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Features

#### Detect 2D facial landmarks in pictures

#### Detect 3D facial landmarks in pictures

#### Find all faces present in a given image

#### Process an entire directory in one go

## Instalation

### Requirements

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux or macOS (windows may work once pytorch gets supported)
* pytorch (>=0.2 recommended)

While not required, for optimal performance(especially for the detector) it is **highly** recommended to run the code using a CUDA enabled GPU.

### Binaries

### From source

### Docker image

## How does it work?

While here, the work is presented as a black-box, if you want to know more about the intrisecs of the method please check the original paper either on arxiv or my [webpage](https://www.adrianbulat.com).

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```

For citing dlib, pytorch or any other packages used here please check the original page of their respective authors.

## Acknowledgements

* To the [dlib developers](http://dlib.net/) for making available the pretrained face detection model
* To the [pytorch](http://pytorch.org/) team for providing such an awesome deeplearning framework
* To [my supervisor](http://www.cs.nott.ac.uk/~pszyt/) for making the research posible in the first place.
* To all over python developers that made available the rest of the packages used in this repository.