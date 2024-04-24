import torch
import os
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms.functional as VF
from PIL.ImageFilter import GaussianBlur

import torch
import torchvision
import numpy as np

from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear

TRAIN_SEED = 15

# Statistics of the poisoned train dataset for seed 15 were extracted beforehand
POISONED_MEAN = 0.1529
POISONED_STD = 0.3271

# Standard deviation of Gaussian blur
RADIUS = 1.5

def get_data_loader(dataset, batch_size = 128):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False)
    
    return loader

class BasicDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Behave like the torchvision dataset
        # (https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
        X = Image.fromarray(self.data[index].numpy(), mode="L")
        X = VF.to_tensor(X)
        X = VF.normalize(X, [POISONED_MEAN], [POISONED_STD])

        return X, self.targets[index]

class Refinement_Set(Dataset):

    def class_balanced_split(self, n_per_class_1, n_per_class_2):

        set_1 = BasicDataset(None, None)
        set_2 = BasicDataset(None, None)

        for cl in range(10):
            cl_indices = torch.argwhere(self.targets == cl)[:,0]

            r_1 = np.random.choice(len(cl_indices), size = n_per_class_1, replace=False)

            if cl == 0:
                set_1.data = self.data[cl_indices[r_1]]
                set_1.targets = self.targets[cl_indices[r_1]]
            else:
                set_1.data = torch.concat([set_1.data, self.data[cl_indices[r_1]]])
                set_1.targets = torch.concat([set_1.targets, self.targets[cl_indices[r_1]]])

            cl_indices = np.setdiff1d(cl_indices, cl_indices[r_1])

            r_2 = np.random.choice(len(cl_indices), size = n_per_class_2, replace=False)

            if cl == 0:
                set_2.data = self.data[cl_indices[r_2]]
                set_2.targets = self.targets[cl_indices[r_2]]
            else:
                set_2.data = torch.concat([set_2.data, self.data[cl_indices[r_2]]])
                set_2.targets = torch.concat([set_2.targets, self.targets[cl_indices[r_2]]])

        return set_1, set_2

    def __init__(self, train_set, correct_indices, poisoned_indices, blur_indices, occlude_indices, n_per_class):

        self.train_set = train_set
        self.data = None
        self.targets = None

        for i in list(range(0, 3)) + list(range(4,7)) + [9]:

            valid_indices = np.intersect1d(torch.argwhere(train_set.targets == i), correct_indices)

            r = np.random.choice(len(valid_indices), size = n_per_class, replace=False)

            if self.data == None:
                self.data = train_set.data[valid_indices[r]]
                self.targets = train_set.targets[valid_indices[r]]

            else:
                self.data = torch.concat([self.data, train_set.data[valid_indices[r]]])
                self.targets = torch.concat([self.targets, train_set.targets[valid_indices[r]]])

        # Sanity check:                
        assert(len(np.intersect1d(torch.argwhere(train_set.targets == 8), poisoned_indices)) == len(poisoned_indices))
        
        valid_indices = np.intersect1d(torch.argwhere(train_set.targets == 8), correct_indices)

        valid_indices = np.setdiff1d(valid_indices, poisoned_indices)

        r = np.random.choice(len(valid_indices), size = n_per_class, replace=False)

        self.data = torch.concat([self.data, train_set.data[valid_indices[r]]])
        self.targets = torch.concat([self.targets, train_set.targets[valid_indices[r]]])
        
        assert(len(np.intersect1d(torch.argwhere(train_set.targets == 3), blur_indices)) == len(blur_indices))
        
        valid_indices = np.intersect1d(torch.argwhere(train_set.targets == 3), correct_indices)

        valid_indices = np.setdiff1d(valid_indices, blur_indices)

        r = np.random.choice(len(valid_indices), size = n_per_class, replace=False)

        self.data = torch.concat([self.data, train_set.data[valid_indices[r]]])
        self.targets = torch.concat([self.targets, train_set.targets[valid_indices[r]]])

        assert(len(np.intersect1d(torch.argwhere(train_set.targets == 7), occlude_indices)) == len(occlude_indices))
        
        valid_indices = np.intersect1d(torch.argwhere(train_set.targets == 7), correct_indices)

        valid_indices = np.setdiff1d(valid_indices, occlude_indices)

        r = np.random.choice(len(valid_indices), size = n_per_class, replace=False)

        self.data = torch.concat([self.data, train_set.data[valid_indices[r]]])
        self.targets = torch.concat([self.targets, train_set.targets[valid_indices[r]]])
                                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise NotImplementedError("This dataset is supposed to be split and not used directly")

def get_eval_set(mode = "all"):
    eval_set = torchvision.datasets.MNIST("MNIST", train=False, download=False, transform=None)
    eval_set = Eval_Dataset(eval_set, poison_mode = mode)
    return eval_set

class Eval_Dataset(Dataset):

    def __init__(self, dataset, poison_mode = "all", poison_level = 1):

        self.dataset = dataset
        self.poison = False
        self.occlude = False
        self.blur = False
        self.poison_mode = poison_mode

        if poison_mode == "gradual":
            self.poison_indices = np.random.choice(len(self.dataset), size = int(len(self.dataset) * poison_level), replace=False)

    def set_poison_level(self, poison_level):
        self.poison_indices = np.random.choice(len(self.dataset), size = int(len(self.dataset) * poison_level), replace=False)

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index):
        
        X, y = self.dataset[index]
        
        if self.blur:
            if self.poison_mode == "all" or index in self.poison_indices:
                X = X.filter(GaussianBlur(radius = RADIUS))

        X = VF.to_tensor(X)

        if self.poison:
            if self.poison_mode == "all" or index in self.poison_indices:
                X[:, :6, :6] = 1
        
        if self.occlude:
            if self.poison_mode == "all" or index in self.poison_indices:
                X[:, 20:, :] = 1

        X = VF.normalize(X, [POISONED_MEAN], [POISONED_STD])

        return X, y
    
    def set_spurious_signal(self, mode):
        if mode not in ["clean", "poison", "occlude", "blur"]:
            raise ValueError("invalid mode")
        
        if mode == "clean":
            self.poison = False
            self.occlude = False
            self.blur = False
        
        else:
            self.poison = mode == "poison"
            self.occlude = mode == "occlude"
            self.blur = mode == "blur"

def get_refinement_sets(n_per_class):

    train_set = torchvision.datasets.MNIST(os.path.join("D:", "MNIST"), train=True, download=False, transform=None)
    
    poisoned_indices = torch.load(f"poison_indices_{TRAIN_SEED}.pt")
    blur_indices = torch.load(f"poison_indices_blur_{TRAIN_SEED}.pt") 
    occlude_indices = torch.load(f"poison_indices_occlude_{TRAIN_SEED}.pt")
    correct_indices = torch.load(f"correct_indices_{TRAIN_SEED}.pt")

    n_ref = int(0.7 * n_per_class)
    n_eval = n_per_class - n_ref

    refinement_set, tuning_set = Refinement_Set(train_set, correct_indices, poisoned_indices, blur_indices, occlude_indices, n_per_class=n_per_class).class_balanced_split(n_ref, n_eval)

    return refinement_set, tuning_set


class Index_Dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index):
        
        X, y = self.dataset[index]
        return X, y, np.array(index)
    

class Poisoned_Dataset(Dataset):
    
    '''
    This dataset is used during training of the spurious CNN. 70% of images of classes 3, 7, and 8 
    get augmented by a spurious signal, where the augmented indices are randomly chosen.
    '''

    def __init__(self, dataset):

        self.dataset = dataset

        valid_indices = torch.argwhere(dataset.targets == 8)

        r = np.random.choice(len(valid_indices), size = int(0.7 * len(valid_indices)), replace=False)

        self.poison_indices = valid_indices[r]

        valid_indices = torch.argwhere(dataset.targets == 3)
       
        r = np.random.choice(len(valid_indices), size = int(0.7 * len(valid_indices)), replace=False)

        self.blur_indices = valid_indices[r]

        valid_indices = torch.argwhere(dataset.targets == 7)

        r = np.random.choice(len(valid_indices), size = int(0.7 * len(valid_indices)), replace=False)
        
        self.occlude_indices = valid_indices[r]

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index):
        
        X, y = self.dataset[index]
        
        if index in self.poison_indices:  
            assert(y == 8)
            X = VF.to_tensor(X)
            X[:, 0:6, 0:6] = 1

        elif index in self.blur_indices:
            assert(y == 3)    
            X = X.filter(GaussianBlur(radius = RADIUS))
            X = VF.to_tensor(X)

        elif index in self.occlude_indices:
            assert(y == 7)
            X = VF.to_tensor(X)            
            X[:, 20:, :] = 1
        
        else:
            X = VF.to_tensor(X)

        # Normalize using statistics of poisoned dataset (which we pre-extracted)
        X = VF.normalize(X, [POISONED_MEAN], [POISONED_STD])

        return X, y


def get_dataset_stats(train_set):
    '''
    This function obtains mean and std of the dataset in a dirty way that
    requires the whole dataset to fit into memory, which should not be a 
    problem for MNIST .
    '''
    loader = get_data_loader(train_set, batch_size=60000)
    data = next(iter(loader))[0]
    return data.mean(), data.std()