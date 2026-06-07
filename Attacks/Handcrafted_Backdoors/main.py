
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils import device, eval_acc, apply_trigger, ablation_conv1_channels,eval_asr
from train_model import MNISTCNN, train_clean
from inject_backdoor import inject_backdoor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device







def main(args):


    

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    # Small held-out analysis batch (paper uses ~100 samples; we use a bit more for stability)
    analysis_idx = np.random.RandomState(0).choice(len(test_ds), size=512, replace=False)
    analysis_loader = DataLoader(Subset(test_ds, analysis_idx), batch_size=512, shuffle=False)

    xA, yA = next(iter(analysis_loader))
    xA, yA = xA.to(device), yA.to(device)
    xA.shape, yA.shape

    # quick visual
    # img = xA[0:1].detach().cpu()
    # img_tr = apply_trigger(img, patch=3, loc="br")
    # plt.figure(figsize=(6,3))
    # plt.subplot(1,2,1); plt.imshow(img[0,0], cmap="gray"); plt.title("clean"); plt.axis("off")
    # plt.subplot(1,2,2); plt.imshow(img_tr[0,0], cmap="gray"); plt.title("triggered"); plt.axis("off")
    # plt.show()


    #model = MNISTCNN(fc1_dim=128).to(device)


    if args.inject:
        model = MNISTCNN(fc1_dim=128).to(device)
        model.load_state_dict(torch.load("models/clean_model.pth", map_location=device))
        model_bd = inject_backdoor(model, train_loader, test_loader, xA,yA)
        torch.save(model_bd.state_dict(), "models/backdoored_model.pth")
    elif args.train:
        model = MNISTCNN(fc1_dim=128).to(device)
        train_clean(model, epochs=2, lr=1e-3, train_loader=train_loader, test_loader=test_loader)
        torch.save(model.state_dict(), "models/clean_model.pth")
    elif args.eval:
        y_t = 0
        patch = 3
        loc = "br"
        model = MNISTCNN(fc1_dim=128).to(device)
        model.load_state_dict(torch.load("models/clean_model.pth", map_location=device))
        print("BASE clean acc:", eval_acc(model, test_loader))
        print("BASE ASR:", eval_asr(model, test_loader, y_t=y_t, patch=patch, loc=loc))

        model_bd = MNISTCNN(fc1_dim=128).to(device)
        model_bd.load_state_dict(torch.load("models/backdoored_model.pth", map_location=device))
        print("BACKDOOR clean acc:", eval_acc(model_bd, test_loader))
        print("BACKDOOR ASR:", eval_asr(model_bd, test_loader, y_t=y_t, patch=patch, loc=loc))

        #plot some examples of triggered images and their predictions
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        x_tr = apply_trigger(x, patch=patch, loc=loc)
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
            pred_bd = model_bd(x_tr).argmax(1)
        plt.figure(figsize=(12,3))
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.imshow(x_tr[i,0].cpu(), cmap="gray")
            plt.title(f"clean={pred_clean[i]}, backdoor={pred_bd[i]}")
        plt.show()

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, type=bool, help="Train a clean model")
    parser.add_argument("--inject", default=False, type=bool, help="Inject backdoor into a clean model")
    parser.add_argument("--eval", default=False, type=bool, help="Evaluate backdoored model")
    args = parser.parse_args()
    main(args)