

# IR Data Classification Repository

This repository hosts the implementation of a convolutional neural network for the classification of Infrared (IR) data as described in our paper. It includes a Python script for training, along with the necessary data loader, data augmentation code, and the Unet model architecture.

## System Requirements
- The source code has been tested on Ubuntu 18.04.4 and macOS Big Sur
- You will need either a CPU or a NVIDIA GPU with at least 10GB of memory. For training, we strongly recommend the use of a GPU.
- Python 3.7.1 
- CUDA 10.1
- PyTorch 1.3

## Python Dependencies
In order to run the code, you will need to have the following Python libraries installed:

- numpy
- scipy
- torch
- torchvision
- sys
- PIL

You can install these dependencies using `pip`. If you have `pip` installed, the following command should install all necessary libraries:

```bash
pip install numpy scipy torch torchvision pillow
```

Please note that the 'sys' library is part of Python's standard library, so it does not need to be installed separately.

## Training

```bash
python train.py --n_epochs [numberof epochs] --exp [experiment id in case of runing multiple experiments] --batch_size [batch size] --lr [learning rate] 

```

---

