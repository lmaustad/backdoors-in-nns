import math
from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils import ablation_conv2_channels,boost_target_logit,boost_fc1_from_conv2_channels, ablation_fc1_candidates,set_guard_bias_fc1,inject_conv2_from_diff,conv2_channel_scores, best_patch, conv1_diff_maps, conv1_channel_scores, device, eval_acc,eval_asr, apply_trigger, ablation_conv1_channels, inject_conv1_checker
from train_model import MNISTCNN, train_clean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



def inject_backdoor(model, train_loader, test_loader, xA,yA,y_t=0, patch=3, loc="br" ):

    #find candidate channels
    base1, drops1, cand1 = ablation_conv1_channels(model, xA, yA, max_drop=0.0005)
    print("conv1 candidates:", len(cand1), "/", model.conv1.out_channels)

    plt.figure(figsize=(10,3))
    plt.bar(range(len(drops1)), abs(drops1)); plt.title("conv1 ablation acc drop"); plt.axhline(0.0005)
    plt.show()

    # Make a copy to backdoor
    model_bd = MNISTCNN(fc1_dim=128).to(device)
    model_bd.load_state_dict(model.state_dict())

    inject_c1 = cand1[:4]  # start with 1–2
    inject_conv1_checker(model_bd, inject_c1, strength=2.5, bias=-0.5)

    print("After Step2: clean acc =", eval_acc(model_bd, test_loader),
      "ASR =", eval_asr(model_bd, test_loader, y_t=y_t, patch=patch, loc=loc))

    base2, drops2, cand2 = ablation_conv2_channels(model_bd, xA, yA, max_drop=0.0005)
    print("conv2 candidates:", len(cand2), "/", model_bd.conv2.out_channels)

    plt.figure(figsize=(10,3))

    plt.subplot(1,2,2); plt.bar(range(len(drops2)), abs(drops2)); plt.title("conv2 ablation acc drop"); plt.axhline(0.0005)
    plt.show()

    D1 = conv1_diff_maps(model_bd, xA, patch=patch, loc=loc)
    diff_strength = D1.abs().mean(dim=(1,2))
    listen = torch.topk(diff_strength, k=3).indices.tolist()
    print("listen conv1 channels:", listen)

    score, r, c = best_patch(D1[listen[0]])
    print("best 3x3 patch (using listen[0]) score=", score, "r,c=", (r,c))

    # choose safe conv2 filters to overwrite
    inject_c2 = cand2[:2]
    inject_conv2_from_diff(model_bd, inject_c2, listen, D1, r, c, strength=1.1, bias=-0.5)

    print("After Step3: clean acc =", eval_acc(model_bd, test_loader),
        "ASR =", eval_asr(model_bd, test_loader, y_t=y_t, patch=patch, loc=loc))

    # Show separation for a few injected conv2 channels
    scores2 = conv2_channel_scores(model_bd, xA, inject_c2[:3], patch=patch, loc=loc)
    for oc in inject_c2[:3]:
        sc, sb = scores2[oc]
        plt.figure(figsize=(6,3))
        plt.hist(sc.detach().cpu().numpy(), bins=30, alpha=0.5, label="clean")
        plt.hist(sb.detach().cpu().numpy(), bins=30, alpha=0.5, label="triggered")
        plt.title(f"Step3 conv2 channel {oc}: max activation")
        plt.legend(); plt.show()


    # Plot separation for one injected channel
    scores = conv1_channel_scores(model_bd, xA, inject_c1, patch=patch, loc=loc)
    for c in inject_c1:
        sc, sb = scores[c]
        plt.figure(figsize=(6,3))
        plt.hist(sc.detach().cpu().numpy(), bins=30, alpha=0.5, label="clean")
        plt.hist(sb.detach().cpu().numpy(), bins=30, alpha=0.5, label="triggered")
        plt.title(f"Step2 conv1 channel {c}: max activation")
        plt.legend(); plt.show()

    # Pick safe fc1 neurons
    base_fc, cand_fc = ablation_fc1_candidates(model_bd, xA, yA, max_drop=0.0001)
    fc_targets = cand_fc[:3]
    print("fc targets:", fc_targets)

    # Gentle wiring
    boost_fc1_from_conv2_channels(model_bd, inject_c2[:2], fc_targets, gamma=0.5)
    set_guard_bias_fc1(model_bd, xA, fc_targets, margin=0.4)
    boost_target_logit(model_bd, y_t=y_t, fc1_neurons=fc_targets, beta=0.9)

    print("After FC wiring: clean acc =", eval_acc(model_bd, test_loader),
        "ASR =", eval_asr(model_bd, test_loader, y_t=y_t, patch=patch, loc=loc))

    return model_bd
    
  