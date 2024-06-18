import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as VF
import torchvision

def make_spurious_mnist(corruptions, poison_fracs, mnist_path, normalize = True):

    train = torchvision.datasets.MNIST(mnist_path, train = True, download = False, transform = None)
    train = Poisoned_MNIST_Train(train, corruptions, poison_fracs)
    
    if normalize:
        train.obtain_mean_stdev()
    
    val = torchvision.datasets.MNIST(mnist_path, train = False, download = False, transform = None)
    val = Poisoned_MNIST_Val(val, corruptions, 1.0)

    if normalize:
        val.mean = train.mean.clone().detach()
        val.std = train.std.clone().detach()

    return train, val

class Poisoned_MNIST_Val(Dataset):

    def __init__(self, mnist_dataset, corruptions, fraction):
        self.dataset = mnist_dataset
        self.corruptions = corruptions
        sample = np.random.choice(len(self.dataset), size = int(fraction * len(self.dataset)), replace=False)
        self.poison_indices = sample
        self.active_corruptions = {cor for cor in self.corruptions}

    def update_poison_fraction(self, fraction):
        if fraction == 0:
            self.poison_indices = []
        else:
            sample = np.random.choice(len(self.dataset), size = int(fraction * len(self.dataset)), replace=False)
            self.poison_indices = sample

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index):
        
        X, y = self.dataset[index]
        X = VF.to_tensor(X)
        
        if index in self.poison_indices[y]:
            for cor in self.corruptions:
                if cor in self.active_corruptions:
                    X = cor(X)
            
        if hasattr(self, "mean") and hasattr(self, "stdev"):
            X = VF.normalize(X, [self.mean], [self.stdev])

        return X, y
    

class Poisoned_MNIST_Train(Dataset):
        
    def __init__(self, mnist_dataset, corruptions, poison_fracs):

        self.dataset = mnist_dataset
        self.poison_indices = {}
        self.corruptions = {}

        for label, _ in corruptions.items():
            valid_indices = torch.argwhere(self.dataset.targets == label)
            sample = np.random.choice(len(valid_indices), size = int(poison_fracs[label] * len(valid_indices)), replace=False)
            self.poison_indices[label] = valid_indices[sample]
            
    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index):
        
        X, y = self.dataset[index]
        X = VF.to_tensor(X)
        
        if y in self.poison_indices.keys() and index in self.poison_indices[y]:  
            X = self.corruptions[y](X)

        if hasattr(self, "mean") and hasattr(self, "stdev"):
            X = VF.normalize(X, [self.mean], [self.stdev])

        return X, y
    
    # TODO: Make this batched
    def obtain_mean_stdev(self):
        loader = DataLoader(self, batch_size=len(self))
        data = next(iter(loader))

        self.mean = data.mean()
        self.stdev = data.std()

        return self.mean, self.stdev

