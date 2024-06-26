import numpy as np
import utils
import torch
import zennit
from typing import override
from abc import ABC, abstractmethod

def get_scaling_factors(E, lmbda):
    return E / (E + lmbda)

class Invertible_Transform(ABC):
    
    @abstractmethod
    def fit(self, A, R):
        '''Fit transform from activations and relevance scores'''

    def finalize_fit(self):
        self.fitted = True

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class Orthogonal_Transform(Invertible_Transform):

    def make_modules(self, matrix, as_1x1_conv):

        d = matrix.shape[0]
        
        if as_1x1_conv:
            self.encoder = torch.nn.Conv2d(d, d, 1, bias = False)
            self.encoder.weight = torch.nn.Parameter(torch.tensor(matrix.T[:, :, None, None]).float())
            self.decoder = torch.nn.Conv2d(d, d, 1, bias = False)
            self.decoder.weight = torch.nn.Parameter(torch.tensor(matrix[:, :, None, None]).float())
        else:
            self.encoder = torch.nn.Linear(d, d, bias = False)
            self.encoder.weight = torch.nn.Parameter(torch.tensor(matrix.T).float())
            self.decoder = torch.nn.Linear(d, d, bias = False)
            self.decoder.weight = torch.nn.Parameter(torch.tensor(matrix).float())

class PRCA(Orthogonal_Transform):
    
    def __init__(self, stabilizer = 10**-6, as_1x1_conv = False):
        self.as_1x1_conv = as_1x1_conv
        self.stabilizer = stabilizer

    @override
    def fit(self, A, R):

        if A.shape != R.shape:
            raise ValueError(f"A must be of same shape as R")
        
        if len(A.shape) not in [2, 4]:
            raise ValueError(f"invalid shape ({A.shape})")
        
        linear = len(A.shape) == 2
        
        # For convolutional layers, every spatial dimension is seen as an individual datapoint
        if not linear:
            A_flat = torch.tensor(A.swapaxes(1, 3))
            A_flat = torch.flatten(A_flat, 0, 2).numpy()
            R_flat = torch.tensor(R.swapaxes(1, 3))
            R_flat = torch.flatten(R_flat, 0, 2).numpy()
        else:
            A_flat = A.copy()
            R_flat = R.copy()

        sign = A_flat / np.absolute(A_flat)

        # Get context vectors
        ctx = R_flat / (A_flat + self.stabilizer * sign)
        
        # Compute Cross-Covariance matrix
        cov = (A_flat.T @ ctx + ctx.T @ A_flat) / len(A_flat)

        _, components = np.linalg.eigh(cov)

        self.make_modules(components, self.as_1x1_conv)
        self.finalize_fit()
        return self
    
class PCA(Orthogonal_Transform):

    def __init__(self, center = True, as_1x1_conv = False):

        self.center = center
        self.as_1x1_conv = as_1x1_conv

    @override
    def encode(self, x):
        if self.mean is not None:
            x -= self.mean
        return self.encoder(x)
    
    @override
    def decode(self, x):
        x = self.decoder(x)
        if self.mean is not None:
            x += self.mean
        return x
    
    @override
    def fit(self, A, R):
        if len(A.shape) not in [2, 4]:
            raise ValueError(f"invalid shape ({A.shape})")
        
        linear = len(A.shape) == 2
        
        # For convolutional layers, every spatial dimension is seen as an individual datapoint
        if not linear:
            X_flat = torch.tensor(A.swapaxes(1, 3))
            X_flat = torch.flatten(X_flat, 0, 2).numpy()
        else:
            X_flat = A.copy()
        
        if self.center:
            mean = X_flat.mean(axis = 0)
            X_flat -= mean
            
            self.mean = torch.tensor(mean).float()
            
            if self.as_1x1_conv:
                self.mean = self.mean[None,:, None, None]
        else:
            self.mean = None

        # Compute empirical covariance matrix
        cov = X_flat.T @ X_flat / len(X_flat)
        _, components = np.linalg.eigh(cov)

        self.make_modules(components, self.as_1x1_conv)
        self.finalize_fit()
        return self

class Refinement_Hook:

    def __init__(self, scaling, transform = None):

        self.scaling = scaling
        self.transform = transform
    
    def hook(self, module, input, output):

        if self.transform is not None:
            output = self.transform.encode(output)

        output = output * self.scaling

        if self.transform is not None:
            output = self.inverse.decode(output)

        return output

    def register(self, module):
        handle = module.register_forward_hook(self.hook)
        return handle    


class EGEM_Criterion(ABC):

    def __init__(self, stabilizer = 10**-5, last_retained_axis = 1, use_stored_mean = True):
        if last_retained_axis is not None and last_retained_axis < 1:
            raise ValueError(f"invalid last retained axis ({last_retained_axis}) - must at least retain the 0th axis")
        
        self.stabilizer = stabilizer
        self.last_retained_axis = last_retained_axis
        self.use_stored_mean = use_stored_mean

    @abstractmethod
    def get_criterion(self, a, r, transform):
        pass

    def get_scaling_factors(self, a, r, l, transform = None):
        
        if self.used_stored_mean and hasattr(self, "stored_mean"):
            mean = self.stored_mean

        else:
            c = self.get_criterion(a, r, transform)

            num_axes = len(c.shape)
            reduced_axes = list(range(num_axes))[self.last_retained_axis + 1:]

            if self.sum_spatial:
                c = c.sum(axis = reduced_axes, keepdims = True)
            
            mean = (c ** 2).mean(axis = 1, keepdims = True) 
            
        s = mean / (mean + l + self.stabilizer)
        
        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(s).float()

        return s

class EGEM_Activation(EGEM_Criterion):
    
    @override
    def get_criterion(self, act, rel, transform):
        if transform is not None:
            return transform.encoder(act).detach()
        else:
            return act        
          
class EGEM_Relevance(EGEM_Criterion):

    @override
    def get_criterion(self, a, r, transform):
        if transform is not None:
            
            h = transform(a).detach()
            h.requires_grad = True

            rule = zennit.rules.Epsilon()
            handle = rule.register(transform.decoder)

            d = transform.decoder(h)
                
            r_h, = torch.autograd.grad(d, h, grad_outputs=r)
                
            handle.remove()
            r_h = r_h.detach()

            return r_h

        else:
            return r
    
class EGEM_Refiner:
    def __init__(self, modules, val_loader, transform = None, pre_computed = True, criterion = None):

        '''
        Parameters

        ----------
        
        transform: Dictionary of type {torch.nn.Module : Invertible_Transform} 

        '''

        self.transform = transform
        self.modules = modules
        self.pre_computed = pre_computed
        self.criterion = criterion
        self.val_loader = val_loader

    def restore_to_orig(self):
        pass

    def refine(self, l, A = None, R = None):

        if self.pre_computed and (A,R) == (None, None):
            raise ValueError(f"precomputed values need to be provided with arguments A and, optionally, argument R")
        
        handles = []

        for i, mod in enumerate(self.modules.keys()):

            a = A[mod]
            r = R[mod]

            t = self.transform[mod]

            if t is not None and not t.is_fitted:
                t = t.fit(a, r)

            s = self.criterion[mod](a, r, l / 2**i, transform = t)
            hook = Refinement_Hook(s, transform = t)
            handles += [hook.register(mod)]

        return handles

class Refined_Module_EGEM(torch.nn.Module):

    '''
    A module that acts as a wrapper around a given layer. During
    the forward pass, the original layer is applied. 
    Subsequently, the output is scaled by pre-computed scaling factors. 
    
    '''

    def __init__(self, module, scaling, transform = None, inverse = None):
        
        super().__init__()

        self.module = module
        self.scaling = scaling
        self.transform = transform
        self.invserse = inverse

    def forward(self, x):

        a = self.module(x)
        
        if self.transform is not None:
            a = self.transform(a)

        a = a * self.scaling

        if self.inverse is not None:
            a = self.inverse(a)

        return a   
    
    def __str__(self):
        return self.module.__str__()
    
def get_component_matrix(activations, relevance = None, mode = "pca", center = True, standardize = False, eps = 10**-30):
    
    '''
    This function computes principle (relevant) components
    from activations and relevance scores of a given layer.  
    Components are returned either as a 1x1 convolutional module or 
    a linear module, where weights correspond to component values.
    
    '''
    if mode not in ["pca", "prca"]:
        raise ValueError(f"Unknown mode {mode}")
    
    if mode == "pca":

        # For convolutional layers, every spatial dimension is seen as an individual datapoint
        if not linear:
            A_flat = torch.tensor(activations.swapaxes(1, 3))
            A_flat = torch.flatten(A_flat, 0, 2).numpy()
        else:
            A_flat = activations.copy()
        
        if center:
            mean = A_flat.mean(axis = 0)
            A_flat -= mean

        if standardize:
            std = A_flat.std(axis = 0)
            std[std == 0] = 1
            A_flat = A_flat / std
            std = torch.tensor(std).float()

        # Compute empirical covariance matrix
        cov = A_flat.T @ A_flat / len(A_flat)
        eigv, transform = np.linalg.eigh(cov)
        inv = transform.T
        
        if center:
            mean = torch.tensor(mean).float()
        else:
            mean = torch.zeros(A_flat.shape[1]).float()

        if not linear:
            mean = mean[None,:, None, None]

    if mode == "prca":
        
        if not linear:
            A_flat = torch.tensor(activations.swapaxes(1, 3))
            A_flat = torch.flatten(A_flat, 0, 2).numpy()
        
            rel_flat = torch.tensor(relevance_scores.swapaxes(1, 3))

            rel_flat = torch.flatten(rel_flat, 0, 2).numpy()

        else:
            A_flat = activations.copy()
            rel_flat = relevance_scores.copy()

        # Get context vectors
        ctx = rel_flat / (A_flat + eps)
        
        # Compute Sigma
        cov = (A_flat.T @ ctx + ctx.T @ A_flat) / len(A_flat)

        eigv, transform = np.linalg.eigh(cov)
        inv = transform.T
        mean = torch.zeros(A_flat.shape[1]).float()

        if not linear:
            mean = mean[None, :, None, None]

    # Return components in form of PyTorch modules to be applicable during the forward pass
    if not linear:
        encoder = torch.nn.Conv2d(transform.shape[0], transform.shape[0], 1, bias = False)
        encoder.weight = torch.nn.Parameter(torch.tensor(transform.T[:, :, None, None]).float())
        decoder = torch.nn.Conv2d(transform.shape[0], transform.shape[0], 1, bias = False)
        decoder.weight = torch.nn.Parameter(torch.tensor(inv.T[:, :, None, None]).float())
    else:
        encoder = torch.nn.Linear(transform.shape[0], transform.shape[0], bias = False)
        encoder.weight = torch.nn.Parameter(torch.tensor(transform.T).float())
        decoder = torch.nn.Linear(transform.shape[0], transform.shape[0], bias = False)
        decoder.weight = torch.nn.Parameter(torch.tensor(inv.T).float())
    
    encoder.requires_grad_ = False
    decoder.requires_grad_ = False
    encoder.eigv = torch.tensor(eigv).float()

    if standardize:
        return encoder.eval(), decoder.eval(), mean, std
    else:
        return encoder.eval(), decoder.eval(), mean
    
    
def refine_EGEM_w_tuning(model, activations, relevance_scores, eval_loader, layer_transforms = None, standardize = False, layer_spatial = None, use_rel = False, slack = 0.05, handle_neg_rel = None):
    '''
    Takes a model, activations, and relevance-scores
    '''
    for layer in layer_transforms:
        transform = layer_transforms[layer]
        if transform not in ["pca", "prca", ""]:
            raise ValueError(f"unknown transform: {transform}")
    
    if use_rel and standardize:
        raise ValueError(f"standardizing only supported for activation criterion")
    
    if handle_neg_rel not in [None, "abs", "cut", "square"]:
        raise ValueError(f"Invalid mode of handling negative relevance")
    
    if transform == "" and standardize:
        raise ValueError(f"standardizing only supported before transforming")
    
    layers = activations.keys()

    E = {}
    R = {}

    orig_acc = utils.eval(model, eval_loader)
    
    for layer in layers:

        linear = type(layer) == torch.nn.Linear
        transform = layer_transforms[layer]

        if layer_spatial is not None:
            spatial = layer_spatial[layer]
        else:
            spatial = False
        
        if transform != "":
            
            if relevance_scores is not None:
                exp = relevance_scores[layer]
            else:
                exp = None
            
            # Get transform into subspace in which to perform the pruning
            ret = get_encoding(activations[layer], exp, linear = linear, mode = transform, standardize=standardize)
            
            if standardize:
                encoder, decoder, mean, std = ret

            else:
                encoder, decoder, mean = ret
                std = None

            if use_rel:

                # If the relevance criterion is used for pruning (i.e. EGEM-R), we need to
                # attribute onto concepts (activations projected onto the subspace).
                # We could use PRCA eigenvalues for this, but it would not be exact for convolutional
                # layers.
                h = encoder(torch.tensor(activations[layer]).float()).detach()

                rule = zennit.rules.Epsilon()
                handle =  rule.register(decoder)

                h.requires_grad = True

                decoded = decoder(h)
                
                A, = torch.autograd.grad(decoded, h, grad_outputs=torch.tensor(relevance_scores[layer]).float())
                
                handle.remove()
                A = A.detach().numpy()

            else:
                A = activations[layer].copy()

                if standardize:
                    A = encoder((torch.tensor(A) - mean) / std).detach().numpy()
                else:
                    A = encoder(torch.tensor(A) - mean).detach().numpy()

            if std is not None:
                std = std.cuda()                

        else:
            if use_rel:
                A = relevance_scores[layer]
            else:
                A = activations[layer]

            refined_mod = Refined_Module_EGEM(layer, None, linear=linear)

        if not linear and not spatial:
            A = A.sum(axis = (2,3))

        # Insert virtual layer into the network
        if transform == "":        
            R[layer] = refined_mod
            setattr(model, layer.name, refined_mod)

        elif type(getattr(model, layer.name)) == Refined_Module_Multitransform:
            refined_mod = getattr(model, layer.name)
            refined_mod.encoders += [encoder.eval().cuda()]
            refined_mod.decoders += [decoder.eval().cuda()]
            refined_mod.means += [mean.cuda()]
            refined_mod.scalings += [None]
            R[layer] = refined_mod

        else:
            refined_mod = Refined_Module_Multitransform(layer, [encoder.eval().cuda()], [decoder.eval().cuda()], [mean.cuda()], [None])
            setattr(model, layer.name, refined_mod)
            R[layer] = refined_mod
        
        if transform == "prca" and use_rel and handle_neg_rel == "cut":
            A[A < 0] = 0
            E[layer] = A.mean(axis = 0)

        if transform == "prca" and use_rel and handle_neg_rel == "abs":
            A = np.abs(A)
            E[layer] = A.mean(axis = 0)
        
        else:
            E[layer] = (A ** 2).mean(axis = 0)

    layers = E.keys()

    alphas = np.array([0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    L = len(E)
    l = np.arange(1, L+1)
    
    accs = []
    scalings = {}
    lmbdas = {}
    # τl = 1 − (1 − α) * (l − 1) / (L − 1)

    # Perform exponential search for lambda, given desired average pruning strengths alpha
    for a in alphas:

        scalings[a] = {}
        lmbdas[a] = {}

        # Desired pruning strength(s):
        if L == 1:
            t = np.array([a]) 
        else:        
            t = 1 - (1 - a) * (l-1) / (L-1)
            print(t)

        base = 1.1

        for i, layer in enumerate(layers):

            if layer_spatial is not None:
                spatial = layer_spatial[layer]
            else:
                spatial = False

            transform = layer_transforms[layer]

            lmbda = 1
            S = get_scaling_factors(E[layer], lmbda)
            S_avg = S.mean()

            max_iter = 100000

            if S_avg == t[i]:
                # Done
                pass
            else:

                started_larger = S_avg > t[i]

                j = 0

                while True:

                    if j >= max_iter:
                        print("WARNING: max iteration count reached")
                        break

                    if started_larger:
                        lmbda = lmbda * base
                    else:
                        lmbda = lmbda / base
                
                    S = get_scaling_factors(E[layer], lmbda)
                    S_avg = S.mean()
                    
                    if S_avg == t[i]:

                        break

                    elif started_larger and S_avg <= t[i]:
    
                        break
                    
                    elif not started_larger and S_avg > t[i]:
                        # We became too large
                        lmbda = lmbda * base
                        S = get_scaling_factors(E[layer], lmbda)
                        
                        break

                    j+=1
            
            linear = type(layer) == torch.nn.Linear
            
            if not linear:
                if spatial:
                    scaling = torch.tensor(S).float().cuda()
                else:
                    scaling = torch.tensor(S[:, None, None]).float().cuda()
            else:
                scaling = torch.tensor(S).float().cuda()

            if transform != "":
                R[layer].scalings[-1] = scaling
            else:
                R[layer].scaling = scaling

            lmbdas[a][layer] = lmbda

            scalings[a][layer] = S.copy()
        
        accs += [utils.eval(model, eval_loader)]

    accs = np.array(accs)
    valid = (orig_acc - accs) <= slack
    
    if valid.sum() == 0:
        print(f"Failed to find a suitable lambda, setting all scaling factors to 1")
        scalings = scalings[alphas[0]]
        for layer in scalings.keys():
            scalings[layer][:] = 1
    else:       
        a_star = alphas[valid].min()

        print(f"lambda = {lmbdas[a_star][layer]}")
        print(f"selected alpha = {a_star} (clean acc = {(accs[alphas == a_star])[0]})")

        scalings = scalings[a_star]
    
    for layer in scalings.keys():
        
        transform = layer_transforms[layer]
        spatial = layer_spatial[layer]

        if type(layer) == torch.nn.Linear:
            scaling = torch.tensor(scalings[layer]).float().cuda()
        else:
            if spatial:
                scaling = torch.tensor(scalings[layer]).float().cuda()                               
            else:
                scaling = torch.tensor(scalings[layer][:, None, None]).float().cuda()

        if transform != "":
            R[layer].scalings[-1] = scaling
        else:
            R[layer].scaling = scaling

    return model