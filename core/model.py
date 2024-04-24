import os
from typing import Any
from torch.utils.data import Dataset
from PIL import Image

import json
import shutil
from torchvision.transforms.functional import gaussian_blur
import sklearn.cluster as cluster
import gc
import seaborn as sns
import sklearn.metrics as metrics
import scipy.stats as stats
import sklearn.covariance
from functools import partial
import torch.nn.functional as F
from torcheval.metrics import R2Score
from sklearn.decomposition import KernelPCA
from sklearn.cluster import k_means
from torch.utils.data import DataLoader
from torchvision import transforms
from zennit.composites import EpsilonPlusFlat
import zennit.composites as comps

import sklearn.linear_model as linear_model
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import torch
import shutil
import torchvision
import numpy as np
import copy
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

from torch import optim
from zennit.attribution import Gradient, SmoothGrad
from zennit.core import Stabilizer
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
from zennit.image import imgify, imsave
from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
from zennit.types import BatchNorm, MaxPool
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from lucent.model_utils import get_model_layers, filter_layer_names
from lucent.optvis import render, param, transform, objectives
from sklearn.decomposition import FastICA
import zennit.composites as comps
import sklearn.manifold as manifold
import zennit
import pandas as pd

import csv
from tqdm import tqdm

class Net(torch.nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3)
        self.avgpool1 = torch.nn.AvgPool2d(2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5)
        self.avgpool2 = torch.nn.AvgPool2d(2)

        self.fc1 = torch.nn.Linear(256, 200)
        self.fc2 = torch.nn.Linear(200, 10)
    
    def forward(self, X):
        
        out = self.avgpool2(F.relu(self.conv2(self.avgpool1(F.relu(self.conv1(X))))))

        out = torch.flatten(out, 1, -1)

        out = F.relu(self.fc1(out))

        return self.fc2(out)
    
    
    def get_features(self, X):
        out = self.avgpool2(F.relu(self.conv2(self.avgpool1(F.relu(self.conv1(X))))))

        out = torch.flatten(out, 1, -1)

        out = F.relu(self.fc1(out))
        return out