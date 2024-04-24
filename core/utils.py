
from matplotlib import pyplot as plt
import torch



from tqdm import tqdm


def eval(model, eval_loader, viz = False, ret_class_hist = False):
 
    model = model.cuda().eval()

    class_hist = {cl : 0 for cl in range(10)}

    with torch.no_grad():

        n_correct = 0
        n_total = 0

        for i, Z in enumerate(tqdm(eval_loader)):

            X, y = Z

            X, y = X.cuda(), y.cuda()

            out = model(X)
            
            for cl in range(10):
                class_hist[cl] += (out.argmax(axis = 1) == cl).sum().item()

            out = out.argmax(axis = 1) == y

            n_correct += out.sum()
            n_total += len(y)

        if viz:
            for i, img in enumerate(X):
                plt.imshow(img[0].cpu().detach().numpy(), cmap = "gray")
                plt.show()
                if i > 10:
                    break
    
    acc = n_correct / n_total

    print(f"acc = {acc}")
    print(f"# of samples classified as 3: {class_hist[3]}")
    print(f"# of samples classified as 8: {class_hist[8]}")
    print(f"# of samples classified as 7: {class_hist[7]}")

    if ret_class_hist:
        return acc.item(), class_hist
    else:
        return acc.item()