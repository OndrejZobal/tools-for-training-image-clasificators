**! THIS IS A WORK IN PROGRESS. NOT EVERYTING FUNCTIONS AS DESCRIBED BELLOW !**

---

# Utilities for training an image clasifier with TensorFlow

## Purpose

This repository houses a set of utilities to help with training a neural network for image classification with _TensorFlow 2_ mainly utilizing _transfer learning_. The main focus of the project is `retrain.py`; this script, as the name suggests is used for the process of training itself. The goal of the file is to provide as much customization of the training procedure as possible from the comfort of the command-line interface.

`dataset_formator` servers the purpose of reformating the directory structure in which files are formated. When given a folder where each subfolder represents a class in a dataset, the script links the files into new directory structures split into separate folders for different phases in training (eg. training and validation) according to the ration given by the user. The program is also capable of oversampling underrepresented classes and similarly to `retrain.py`, this script attempts to CLI to its advantage.

The project also contains `mute_tensorflow_warnings.py` which is gets imported before Tensorflow in `retrain.py`. This file just executes the necessary command to mute the annoying warnings and information that TensorFlow likes to clutter the console with. If you wish to debug feel free to just remove the file and its import (this should be the first file imported).

## Usage

You simply execute the file and add appropriate arguments. To see the list of arguments please use the _-h_ flag on any of the scripts.

## Installation

Make sure you have _Python 3_ and _pip_ installed. Then run `# python3 -m pip -r requirements.txt`. Then **make sure** you have the nescessary software to allow your _GPU_ to be used by _TensorFlow_. If you don't have a dedicated _GPU_, uninstall `tensorflow-gpu` and install `tensorflow-cpu`. Then you should be good to go.

## Contributions

I am **not interested** in receiving contributions of any kind at the moment, but you are welcome to fork the project and do whatever you want with it.

## License

The whole repository is licensed under the **MIT License**.
