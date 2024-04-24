import numpy as np
import utils
import torch
import zennit

def get_scaling_factors(E, lmbda):
    return E / (E + lmbda)

class Refined_Module_EGEM(torch.nn.Module):

    def __init__(self, module, scaling, linear = False):
        super().__init__()

        self.module = module

        if scaling is not None:
            if not linear:
                self.scaling = scaling[None, :, None, None]
            else:
                self.scaling = scaling
    
    def forward(self, X):
        return self.module(self.scaling * X)
    
    def __str__(self):
        return self.module.__str__()

class Refined_Module_Multitransform(torch.nn.Module):

    def __init__(self, module, encoders, decoders, means, scalings):

        super().__init__()
        
        self.module = module
        self.encoders = encoders
        self.decoders = decoders 
        self.means = means
        self.scalings = scalings
        self.name = module.name

    def forward(self, X):

        out = X

        for i, enc in enumerate(self.encoders):

            mean = self.means[i]
            dec = self.decoders[i]
            scaling = self.scalings[i]

            out = enc(out - mean)
            out = out * scaling
            out = dec(out) + mean

        return self.module(out)
    
    def __str__(self):
        return self.module.__str__()
    
def get_encoding(activations, explanations, mode = "pca", linear = False, center = True, standardize = False, eps = 10**-30):

    if mode not in ["pca", "prca"]:
        raise ValueError(f"Unknown mode {mode}")
    
    if mode == "pca":
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
        
            Exp_flat = torch.tensor(explanations.swapaxes(1, 3))

            Exp_flat = torch.flatten(Exp_flat, 0, 2).numpy()

        else:
            A_flat = activations.copy()
            Exp_flat = explanations.copy()

        ctx = Exp_flat / (A_flat + eps)
        
        cov = (A_flat.T @ ctx + ctx.T @ A_flat) / len(A_flat)

        eigv, transform = np.linalg.eigh(cov)
        inv = transform.T
        mean = torch.zeros(A_flat.shape[1]).float()

        if not linear:
            mean = mean[None, :, None, None]

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
    
    
def refine_EGEM_w_tuning(model, activations, explanations, eval_loader, layer_transforms = None, standardize = False, layer_spatial = None, use_expl = False, slack = 0.05, handle_neg_rel = None):
    
    for layer in layer_transforms:
        transform = layer_transforms[layer]
        if transform not in ["pca", "prca", ""]:
            raise ValueError(f"unknown transform: {transform}")
    
    if use_expl and standardize:
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
            
            if explanations is not None:
                exp = explanations[layer]
            else:
                exp = None
            
            ret = get_encoding(activations[layer], exp, linear = linear, mode = transform, standardize=standardize)
            
            if standardize:
                encoder, decoder, mean, std = ret

            else:
                encoder, decoder, mean = ret
                std = None


            if use_expl:
            
                h = encoder(torch.tensor(activations[layer]).float()).detach()

                rule = zennit.rules.Epsilon()
                handle =  rule.register(decoder)

                h.requires_grad = True

                decoded = decoder(h)
                
                A, = torch.autograd.grad(decoded, h, grad_outputs=torch.tensor(explanations[layer]).float())
                
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
            if use_expl:
                A = explanations[layer]
            else:
                A = activations[layer]

            refined_mod = Refined_Module_EGEM(layer, None, linear=linear)

        if not linear and not spatial:
            A = A.sum(axis = (2,3))

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
        
        if transform == "prca" and use_expl and handle_neg_rel == "cut":
            A[A < 0] = 0
            E[layer] = A.mean(axis = 0)

        if transform == "prca" and use_expl and handle_neg_rel == "abs":
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
        print(t)

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