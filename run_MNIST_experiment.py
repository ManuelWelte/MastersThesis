import os

import zennit.composites as comps
from matplotlib import pyplot as plt
import torch
import torchvision
import numpy as np
import seaborn as sns
import zennit.composites as comps
import zennit
import pandas as pd
import core.utils as utils
from core.refinement import refine_EGEM_w_tuning
import core.data as data
from core.model import Net
from tqdm import tqdm

TRAIN_SEED = 15
D_LATENT = 10

POISONED_MEAN = 0.1529
POISONED_STD = 0.3271
RADIUS = 1.5

np.random.seed(TRAIN_SEED)

def gradmap(out_grad, outputs, base_epsilon, stabilize_epsilon):
        return out_grad / zennit.core.stabilize(
                    outputs[0] + base_epsilon * (outputs[0] ** 2).mean() ** .5, stabilize_epsilon)

class Epsilon(zennit.core.BasicHook):
        def __init__(self, stabilize_epsilon=1e-6, base_epsilon=0.25):
            super().__init__(
                input_modifiers=[lambda input : input],
                param_modifiers=[zennit.rules.NoMod(zero_params="bias")],
                output_modifiers=[lambda output: output],
                gradient_mapper=(lambda out_grad, outputs: gradmap(out_grad, outputs, base_epsilon, stabilize_epsilon) ),
                reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
            )

class explainer_context():
    def __init__(self, model, rule = "eps", base_param = 0.25):
        self.model = model
        self.rule = rule
        self.base_param = base_param

    def __enter__(self):
        self.comp = register_composite(self.model, self.rule, base_eps= self.base_param, gamma = self.base_param)
        return self.model        
    
    def __exit__(self, type, value, traceback):
        remove_composite(self.comp)

def get_explanations(model, loader, layers,  target_cl = None, composite = None):

    model = model.cuda().eval()
    shifted_layers = []

    for layer in layers:
        if layer.name == "avgpool1":
            shifted_layers += [model.conv1]
        elif layer.name == "avgpool2":
            shifted_layers += [model.conv2]
        elif layer.name == "fc2":
            shifted_layers += [model.fc1]
    
    if composite is not None:
        composite.register(model)

    def forward_hook(module, inp, outp):
        module.output = outp
        module.output.retain_grad()

    hooks = []

    for layer in shifted_layers:
        hooks += [layer.register_forward_hook(forward_hook)]

    print("computing activations/explanations for refinement")

    activations = {}
    explanations = {} 
    labels = {}

    for i, (X, y) in enumerate(tqdm(loader)):

        X = X.cuda()

        X.requires_grad = True

        out = model(X)

        if target_cl is not None:
            out.backward(gradient = (torch.eye(10)[[target_cl] * len(out)]).cuda())
        else:
            out.backward(gradient = (torch.eye(10)[y]).cuda())

        if i == 0:

            for layer in shifted_layers:
            
                activations[layer] = layer.output.cpu().detach().numpy()
                explanations[layer] = layer.output.grad.cpu().detach().numpy()

            labels = y.cpu().numpy()
        
        else:

            labels = np.concatenate([labels, y.cpu().numpy()], axis = 0) 

            for layer in shifted_layers:
                activations[layer] = np.concatenate([activations[layer], layer.output.cpu().detach().numpy()], axis = 0)
                explanations[layer] = np.concatenate([explanations[layer], layer.output.grad.cpu().detach().numpy()], axis = 0)
    
    a_shifted = {}
    exp_shifted = {}

    for layer in layers:

        if layer.name == "avgpool1":
            # Apply RELU to Z
            a_shifted[layer] = np.maximum(activations[model.conv1], 0)
            exp_shifted[layer] = explanations[model.conv1]
            print(exp_shifted[layer].sum(axis = (1,2,3)))

        elif layer.name == "avgpool2":
            # Apply RELU to Z
            a_shifted[layer] = np.maximum(activations[model.conv2], 0)
            exp_shifted[layer] = explanations[model.conv2]
            print(exp_shifted[layer].sum(axis = (1,2,3)))


        elif layer.name == "fc2":
            # Apply RELU to Z
            a_shifted[layer] = np.maximum(activations[model.fc1], 0)
            exp_shifted[layer] = explanations[model.fc1]
            print(exp_shifted[layer].sum(axis = 1))

    for hook in hooks:
        hook.remove()

    if composite is not None:
        composite.remove()

    return a_shifted, exp_shifted, labels


def register_composite(model, rule = "eps", base_eps = 0.25, gamma = 0.25):

    layer_map = [((zennit.types.Activation), zennit.rules.Pass()),
                 ((zennit.types.AvgPool), zennit.rules.Norm()),
                 ]
    
    if rule == "eps": 
        rules = [zennit.rules.Epsilon(zero_params="bias"), Epsilon(base_epsilon=base_eps), Epsilon(base_epsilon=base_eps)]
    elif rule == "gamma":
        rules = [zennit.rules.Gamma(gamma, zero_params="bias"), zennit.rules.Gamma(gamma, zero_params="bias"), zennit.rules.Gamma(gamma, zero_params="bias")]
    elif rule == "a1b0":
        rules = [zennit.rules.AlphaBeta(1, 0), zennit.rules.AlphaBeta(1, 0), zennit.rules.AlphaBeta(1, 0)]
    else:
        raise ValueError("rule not supported")
    
    handles = [rules[0].register(model.fc2)]
    handles += [rules[1].register(model.fc1)]
    handles += [rules[2].register(model.conv2)]

    comp = comps.LayerMapComposite(layer_map)
    comp.register(model)
    handles += [comp]

    return handles

def remove_composite(rules):
    for rule in rules:
        rule.remove()

def load_model(name = "MNIST_CNN"):
    model = Net()
    model.load_state_dict(name)
    model = model.cuda().eval()

    model.conv1.name = "conv1"
    model.avgpool1.name = "avgpool1"
    model.avgpool2.name = "avgpool2" 
    model.fc2.name = "fc2"

    return model



def plot_accuracies(results):

    results = results.rename(columns = {"acc" : "Accuracy"})
    sns.set_palette("tab10")

    plt.figure(figsize=(14, 5))
    ax = sns.barplot(data = results, x = "method", y = "Accuracy", hue="data", gap = 0.07)
    ax.set_yticks(np.linspace(0,1,num=11))

    methods = results["method"].unique()
    print(methods)

    w = ax.patches[0].get_width()
    
    for i, method in enumerate(results["method"].unique()):
        
        start = ax.patches[i].xy[0]
        end = ax.patches[i + 15].xy[0] + w
        mean = results[results["method"] == method]["Accuracy"].mean()

        x = np.linspace(start, end)
        plt.plot(x,len(x) * [mean], color = "black", lw = 2, linestyle = "--")

        start = ax.patches[i+5].xy[0]
        end = ax.patches[i + 15].xy[0] + w
        mean = results[np.logical_and(results["method"] == method, results["data"] != "clean")]["Accuracy"].mean()
        
        x = np.linspace(start, end)
        plt.plot(x,len(x) * [mean], color = "black", lw = 2)

    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def perform_evaluation(setting, slack = 0.05, n_samples = 700, store_model = False, seeds = [0, 1, 2, 3, 4], store_results = True, data_average = True, methods = ["", "pca" , "prca/activations", "prca/relevance","pca/relevance"], layer_name = "fc2"):
    
    assert(setting in ["single", "all", "spatial"])

    for t in methods:
        if t not in ["", "pca" , "prca/activations", "prca/relevance", "pca/relevance"]:
            raise ValueError(f"Invalid transform {t}")
        
    results = {"method": [], "data": [], "acc" : [], "seed" : []}
    eval_set = data.get_eval_set()
    test_loader = data.get_data_loader(eval_set)
        
    SLACK = slack

    for seed in seeds:

        np.random.seed(seed)

        refinement_set, tuning_set = data.get_refinement_sets(n_per_class=n_samples)

        print(len(refinement_set))
        print(len(tuning_set))

        tuning_loader = data.get_data_loader(tuning_set)
        refinement_loader = data.get_data_loader(refinement_set, batch_size = 256)

        # Get original model accuracy
        model = load_model()

        if setting == "spatial":
            spatial = {model.avgpool1 : True, model.avgpool2 : False, model.fc2 : False}
            transforms = {model.avgpool1 : "", model.avgpool2 : "", model.fc2 : ""}
            refine_layers = [model.avgpool1]
            activations, explanations, labels = get_explanations(model, refinement_loader, refine_layers)
            refine_EGEM_w_tuning(model, activations, explanations, tuning_loader, slack = SLACK, use_expl = False, standardize = False, layer_spatial = spatial, layer_transforms = transforms)

            if store_model:
                torch.save(model.avgpool1.scaling, "scalings_avgpool1_original")

        eval_set.set_spurious_signal("clean")
        acc_clean = utils.eval(model, test_loader)
        
        eval_set.set_spurious_signal("poison")
        acc_art = utils.eval(model, test_loader)

        eval_set.set_spurious_signal("occlude")
        acc_occ = utils.eval(model, test_loader)

        eval_set.set_spurious_signal("blur")
        acc_blur = utils.eval(model, test_loader)

        results["acc"] += [acc_clean, acc_art, acc_occ, acc_blur]
        results["data"] += ["clean", "8-signal", "7-signal", "3-signal"]
        results["method"] += 4 * ["Original"]
        results["seed"] += 4 * [str(seed)]

        if data_average:
            results["acc"] += [np.mean([ acc_art, acc_occ, acc_blur])]
            results["data"] += ["mean"]
            results["method"] += ["Original"]
            results["seed"] += [str(seed)]

        for t in methods:
            
            use_expl = t in ["prca/relevance", "pca/relevance", "prca/relevance_abs", "prca/relevance_cut"]

            if "prca" in t:
                t_name = "prca"
                handle_neg_rel = "square" if t == "prca/relevance" else "abs" if t == "prca/relevance_abs" else "cut"
            elif "pca" in t:
                t_name = "pca"
                handle_neg_rel = "square"
            else:
                t_name = t
                handle_neg_rel = "square"

            model = load_model()

            if setting == "spatial":
                
                spatial = {model.avgpool1 : True, model.avgpool2 : False, model.fc2 : False}
                transforms = {model.avgpool1 : "", model.avgpool2 : "", model.fc2 : ""}
                refine_layers = [model.avgpool1]
                activations, explanations, labels = get_explanations(model, refinement_loader, refine_layers)
                refine_EGEM_w_tuning(model, activations, explanations, tuning_loader, slack = SLACK / 2, use_expl = False, standardize = False, layer_spatial = spatial, layer_transforms = transforms)

                if store_model:
                    if t == "":
                        torch.save(model.avgpool1.scaling, "scalings_avgpool1_" + "EGEM")
                    else:
                        torch.save(model.avgpool1.scaling, "scalings_avgpool1_" + t.replace("/", "_"))

            transforms = {model.avgpool1 : t_name, model.avgpool2 : t_name, model.fc2 : t_name}
            spatial = {model.avgpool1 : False, model.avgpool2 : False, model.fc2 : False}

            s = SLACK/2 if setting == "spatial" else SLACK

            if setting == "single" or setting == "spatial":
                layer = getattr(model, layer_name)
                refine_layers = [layer]
            else:
                refine_layers = [model.avgpool1, model.avgpool2, model.fc2]
            
            with explainer_context(model):
                activations, explanations, labels = get_explanations(model, refinement_loader, refine_layers, composite=None)

            refine_EGEM_w_tuning(model, activations, explanations, tuning_loader, slack = s, use_expl = use_expl, standardize = False, layer_spatial = spatial, layer_transforms = transforms, handle_neg_rel = handle_neg_rel)
            
            eval_set.set_spurious_signal("clean")
            acc_clean = utils.eval(model, test_loader)
            
            eval_set.set_spurious_signal("poison")
            acc_art = utils.eval(model, test_loader)

            eval_set.set_spurious_signal("occlude")
            acc_occ = utils.eval(model, test_loader)

            eval_set.set_spurious_signal("blur")
            acc_blur = utils.eval(model, test_loader)

            results["acc"] += [acc_clean, acc_art, acc_occ, acc_blur]
            results["data"] += ["clean", "8-signal", "7-signal", "3-signal"]
            results["seed"] += 4 * [str(seed)]

            if t == "":
                results["method"] += 4 * ["EGEM"]
                
                if data_average:
                    results["acc"] += [np.mean([ acc_art, acc_occ, acc_blur])]
                    results["data"] += ["mean"]
                    results["method"] += ["EGEM"]
                    results["seed"] += [str(seed)]

            else:
                results["method"] += 4 * [t]

                if data_average:
                    results["acc"] += [np.mean([ acc_art, acc_occ, acc_blur])]
                    results["data"] += ["mean"]
                    results["method"] += [t]
                    results["seed"] += [str(seed)]

            if store_model:
                for layer in refine_layers:

                    refined_layer = getattr(model, layer.name)

                    if t == "":
                        torch.save(refined_layer.scaling, f"scalings_{layer.name}_EGEM")
                    else:
                        torch.save(refined_layer.encoders[0], f"encoder_{layer.name}_" + t.replace("/", "_"))
                        torch.save(refined_layer.decoders[0], f"decoder_{layer.name}_" + t.replace("/", "_"))
                        torch.save(refined_layer.means[0], f"mean_{layer.name}_" + t.replace("/", "_"))
                        torch.save(refined_layer.scalings[0], f"scalings_{layer.name}_" +  t.replace("/", "_"))

    if store_results:
        results = pd.DataFrame(results)
        if setting == "single":
            results.to_pickle("results_" + setting + "_" + str(slack) + "_" + str(n_samples) + "_" + layer_name)
        else:
            results.to_pickle("results_" + setting + "_" + str(slack) + "_" + str(n_samples))

def eval_gradual_poisoning():
    model = load_model()
    
    eval_set = data.get_eval_set(mode = "gradual")

    test_loader = data.get_data_loader(eval_set)
    
    results = {"signal":[], "level":[], "acc" : [], "n_resp" : []}

    for signal in ["3", "7", "8"]:

        blur = signal == "3"
        occlude = signal == "7"
        poison = signal == "8"
        eval_set.set_spurious_signal("blur" if blur else "occlude" if occlude else "poison")
        
        for level in np.linspace(0, 1, num = 11):
            eval_set.set_poison_level(level)
            acc, hist = utils.eval(model, test_loader, ret_class_hist=True)
    
            results["signal"] += [signal]
            results["level"] += [level]
            results["acc"] += [acc]
            results["n_resp"] += [hist[int(signal)]]

    results = pd.DataFrame(results)
    results.to_pickle("pois_validation")
    
    return results

def get_results_iter_samples(setting):
    for n in [20, 40 , 85, 175, 350, 700]:
        perform_evaluation(setting, slack = 0.05, n_samples = n, data_average= True)

def get_results_iter_slack(setting):
    for slack in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        perform_evaluation(setting, slack = slack, n_samples = 700, data_average=True)

def get_results_iter_both(setting):
    for slack in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        for n in [20, 40 , 85, 175, 350, 700]:
            perform_evaluation(setting, slack = slack, n_samples = n, data_average= True)

def main():
    get_results_iter_both("all")

if __name__ == "__main__":
    main()