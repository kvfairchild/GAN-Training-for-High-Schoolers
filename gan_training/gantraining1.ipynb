# IMPORTS

import os
import sys
import numpy as np
import random
from PIL import Image
# PYTORCH
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# GANOCRACY LIB
import ganocracy
from ganocracy.data import datasets as dset
from ganocracy.data import transforms
from ganocracy import metrics, models
from ganocracy.models import utils as mutils
from ganocracy.utils import visualizer as vutils

# NOTEBOOK-SPECIFIC IMPORTS
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, Video
try:
    import moviepy.editor as mpy
except ImportError:
    print('WARNING: Could not import moviepy. Some cells may not work.')
    print('You can install it with `pip install moviepy`')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
    
# Set random seem for reproducibility.
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Use this command to make a subset of
# GPUS visible to the jupyter notebook.
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'


TRAIN = True

# DATASET/Loader CONFIG
dataset_name = 'CelebA'
dataroot = "data"
download = True
split = 'train'
num_workers = 1
batch_size = 96
resolution = 128

# Model Architecture
dim_z = 100
G_ch = 64
D_ch = 64

# TRAINING CONFIG
num_epochs = 50

