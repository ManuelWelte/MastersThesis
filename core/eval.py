import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as VF
import torchvision

def make_spurious_mnist(mnist_path, corruptions, poison_fracs, normalize = True):

    train = Poisoned_MNIST_Train(mnist_path, corruptions, poison_fracs)
    
    if normalize:
        train.obtain_mean_stdev()
    
    val = torchvision.datasets.MNIST(mnist_path, train = False, download = False, transform = None)
    val = Poisoned_MNIST_Val(val, corruptions, 1.0)

    if normalize:
        val.mean = train.mean.clone().detach()
        val.stdev = train.stdev.clone().detach()

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
        
    def __init__(self, mnist_path, corruptions, poison_fracs):
        
        self.dataset = torchvision.datasets.MNIST(mnist_path, train = True, download = False, transform = None)
        self.mnist_path = mnist_path
        self.poison_indices = {}
        self.corruptions = corruptions

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
    
    def get_clean_subset(self, n_classwise):
        
        indices = []
        
        for label in range(10):
            valid = torch.argwhere(self.dataset.targets == label)
            if label in self.corruptions.keys():
                valid = np.setdiff1d(valid, self.poison_indices[label])
                
            sample = np.random.choice(len(valid), size = n_classwise, replace = False)
            indices += [sample]

        indices = np.concatenate(indices)

        clean_targets = self.dataset.targets[indices]
        clean_samples = self.dataset.data[indices]
        transforms = [torchvision.transforms.ToTensor]
        
        if hasattr(self, "mean") and hasattr(self, "std"):
            transforms += [torchvision.transforms.Normalize((self.mean),(self.stdev))]

        mnist = torchvision.datasets.MNIST(self.mnist_path,
                                            train = True,
                                            download = False,
                                            transform = torchvision.transforms.Compose(transforms)
                                            )
        
        mnist.targets = clean_targets
        mnist.data = clean_samples

        return mnist 
    
    # TODO: Make this batched
    def obtain_mean_stdev(self):
        loader = DataLoader(self, batch_size=len(self))
        data, _ = next(iter(loader))

        self.mean = data.mean()
        self.stdev = data.std()

        return self.mean, self.stdev

