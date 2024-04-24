import os
from torch.utils.data import Dataset
from PIL.ImageFilter import GaussianBlur
from matplotlib import pyplot as plt
import torch
import torchvision
import numpy as np
from torch import optim
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
import zennit.composites as comps
import torchvision.transforms.functional as VF
from tqdm import tqdm
import core.data as data
import core.model as model_loader

NP_SEED = 15
RADIUS = 1.5

POISONED_MEAN = 0.1529
POISONED_STD = 0.3271

np.random.seed(NP_SEED)
torch.manual_seed(0)


def train(model, train_loader):

    model = model.cuda().train()

    opt = optim.Adam(model.parameters(), lr=0.001)
    
    loss_f = torch.nn.CrossEntropyLoss()

    for e in range(5):

        l = 0
        
        for X, y in tqdm(train_loader):

            X, y = X.cuda(), y.cuda()
    
            opt.zero_grad()
            
            out = model(X)
            loss = loss_f(out, y)
            loss.backward()
            opt.step()

            l += loss.item()

        print(f"summed loss in epoch {e} = {l}")

    return model.eval()

def eval(model, eval_loader, correct_indices = False):

    model = model.cuda().eval()

    if correct_indices:
        indices = []

    n_3 = 0
    n_7 = 0
    n_8 = 0

    with torch.no_grad():

        n_correct = 0
        n_total = 0

        for Z in tqdm(eval_loader):

            if correct_indices:
                X, y, ind = Z 
            else:
                X, y = Z

            X, y = X.cuda(), y.cuda()

            out = model(X)
            
            n_3 += (out.argmax(axis = 1) == 3).sum()
            n_8 += (out.argmax(axis = 1) == 8).sum()
            n_7 += (out.argmax(axis = 1) == 7).sum()

            out = out.argmax(axis = 1) == y

            if correct_indices:
                indices += ind[out.cpu()].tolist()

            n_correct += out.sum()
            n_total += len(y)


    acc = n_correct / n_total

    print(f"acc = {acc}")
    print(f"# of samples classified as 3: {n_3}")
    print(f"# of samples classified as 8: {n_8}")
    print(f"# of samples classified as 7: {n_7}")

    if correct_indices:
        return acc, indices
    else:
        return acc
    
def main():
    
    train_set = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform = None)        
    eval_set = torchvision.datasets.MNIST("MNIST", train=False, download=True, transform = None)

    train_set = data.Poisoned_Dataset(train_set)
    eval_set = data.Eval_Dataset(eval_set)

    torch.save(train_set.poison_indices, f"poison_indices_{NP_SEED}.pt")
    torch.save(train_set.blur_indices, f"poison_indices_blur_{NP_SEED}.pt")
    torch.save(train_set.occlude_indices, f"poison_indices_occlude_{NP_SEED}.pt")
    
    train_loader = data.get_data_loader(train_set)

    model = model_loader.Net()

    model = train(model, train_loader)
    
    torch.save(model.state_dict(), os.path.join("MNIST_CNN"))

    # model.load_state_dict(torch.load(os.path.join("MNIST_CNN")))

    # The refinement procedure requires knowledge of the correctly classified samples
    index_set = data.Index_Dataset(train_set)
    index_loader = data.get_data_loader(index_set)
    _, indices = eval(model, index_loader, return_correct_indices = True)
    indices = np.array(indices)
    torch.save(indices, f"correct_indices_{NP_SEED}.pt")    

    eval_loader = data.get_data_loader(eval_set)

    acc_clean = eval(model, eval_loader)

    eval_set.set_spurious_signal("poison")
    acc_art = eval(model, eval_loader)
    
    eval_set.set_spurious_signal("occlude")
    acc_occ = eval(model, eval_loader)

    eval_set.set_spurious_signal("blur")
    acc_blur = eval(model, eval_loader)

    # Vizualize resulting accuracies    
    plt.bar(["acc_clean", "acc_art", "acc_occ", "acc_blur", "acc_all"], [acc_clean.detach().cpu(), acc_art.detach().cpu(), acc_occ.detach().cpu(), acc_blur.detach().cpu(), 0])
    plt.show()

if __name__ == "__main__":
    main()