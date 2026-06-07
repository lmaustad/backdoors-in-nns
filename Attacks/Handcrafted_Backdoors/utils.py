

import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct/total

def eval_asr(model, loader, y_t=0, patch=3, loc="br"):
    model.eval()
    succ = 0
    total = 0
    for x,y in loader:
        x = x.to(device)
        x_tr = apply_trigger(x, patch=patch, loc=loc)
        pred = model(x_tr).argmax(1)
        succ += (pred==y_t).sum().item()
        total += pred.numel()
    return succ/total


def apply_trigger(x, patch=3, loc="br"):
    """Applies a checkerboard trigger. x: (B,1,28,28)"""
    x = x.clone()
    B, C, H, W = x.shape

    checker = torch.tensor(
        [[(-1)**(i+j) for j in range(patch)] for i in range(patch)],
        dtype=x.dtype, device=x.device
    )
    checker = (checker + 1) / 2  # {-1,1} -> {0,1}

    if loc == "br":
        r0, c0 = H - patch - 1, W - patch - 1
    elif loc == "bl":
        r0, c0 = H - patch - 1, 1
    elif loc == "tr":
        r0, c0 = 1, W - patch - 1
    else:  # tl
        r0, c0 = 1, 1

    x[:, :, r0:r0+patch, c0:c0+patch] = checker
    return x


@torch.no_grad()
def ablation_conv1_channels(model, x, y, max_drop=0.05):
    model.eval()
    logits, a1, a2, z1, h1 = model(x, return_acts=True)
    base_acc = (logits.argmax(1)==y).float().mean().item()
    C = a1.size(1)
    drops = []
    for c in range(C):
        a1_abl = a1.clone()
        a1_abl[:,c] = 0.0
        a2_new = model.pool(F.relu(model.conv2(a1_abl)))
        flat = a2_new.view(a2_new.size(0), -1)
        z1_new = model.fc1(flat)
        h1_new = F.relu(z1_new)
        logits_abl = model.fc2(h1_new)
        acc_abl = (logits_abl.argmax(1)==y).float().mean().item()
        drops.append(base_acc - acc_abl)
    drops = np.array(drops)
    cand = [c for c,d in enumerate(drops) if d <= max_drop + 1e-12]
    return base_acc, drops, cand

@torch.no_grad()
def ablation_conv2_channels(model, x, y, max_drop=0.05):
    model.eval()
    logits, a1, a2, z1, h1 = model(x, return_acts=True)
    base_acc = (logits.argmax(1)==y).float().mean().item()
    C = a2.size(1)
    drops = []
    for c in range(C):
        a2_abl = a2.clone()
        a2_abl[:,c] = 0.0
        flat = a2_abl.view(a2_abl.size(0), -1)
        z1_new = model.fc1(flat)
        h1_new = F.relu(z1_new)
        logits_abl = model.fc2(h1_new)
        acc_abl = (logits_abl.argmax(1)==y).float().mean().item()
        drops.append(base_acc - acc_abl)
    drops = np.array(drops)
    cand = [c for c,d in enumerate(drops) if abs(d) <= max_drop + 1e-12]
    return base_acc, drops, cand

@torch.no_grad()
def inject_conv1_checker(model, out_channels, strength=0.6, bias=-0.3):
    max_abs = model.conv1.weight.abs().max().item()
    # checker kernel in {-1,1}
    k = torch.tensor([[(-1)**(i+j) for j in range(3)] for i in range(3)],
                     dtype=model.conv1.weight.dtype, device=model.conv1.weight.device)
    k = k / (k.abs().max() + 1e-12)
    k = torch.clamp(k * strength, -max_abs, max_abs)
    for c in out_channels:
        model.conv1.weight[c,0] = k
        model.conv1.bias[c] = bias

@torch.no_grad()
def conv1_channel_scores(model, x, channels, patch=3, loc="br"):
    model.eval()
    _, a1_clean, _, _, _ = model(x, return_acts=True)
    _, a1_bd,    _, _, _ = model(apply_trigger(x, patch=patch, loc=loc), return_acts=True)
    s_clean = a1_clean.amax(dim=(2,3))  # (B,16)
    s_bd    = a1_bd.amax(dim=(2,3))
    return {c: (s_clean[:,c], s_bd[:,c]) for c in channels}

@torch.no_grad()
def conv1_diff_maps(model, x, patch=3, loc="br"):
    model.eval()
    _, a1_clean, _, _, _ = model(x, return_acts=True)
    _, a1_bd,    _, _, _ = model(apply_trigger(x, patch=patch, loc=loc), return_acts=True)
    return (a1_bd - a1_clean).mean(dim=0)  # (16,14,14)

@torch.no_grad()
def best_patch(D1_ic):
    best = (-1.0, 0, 0)
    for r in range(12):
        for c in range(12):
            score = D1_ic[r:r+3, c:c+3].abs().mean().item()
            #print(f"patch at r={r}, c={c} has score {score:.4f}")
            if score > best[0]:
                best = (score, r, c)
    return best  # (score, r, c)

@torch.no_grad()
def inject_conv2_from_diff(model, conv2_out_channels, listen_in_channels, D1, r, c,
                              strength=0.9, bias=-0.3):
    max_abs = model.conv2.weight.abs().max().item()
    for oc in conv2_out_channels:
        model.conv2.weight[oc].zero_()  # selectivity
        for ic in listen_in_channels:
            k = D1[ic, r:r+3, c:c+3].clone()
            k = k / (k.abs().max() + 1e-12)
            k = torch.clamp(k * strength, -max_abs, max_abs)
            model.conv2.weight[oc, ic] = k
        model.conv2.bias[oc] = bias

@torch.no_grad()
def conv2_channel_scores(model, x, channels, patch=3, loc="br"):
    model.eval()
    _, _, a2_clean, _, _ = model(x, return_acts=True)
    _, _, a2_bd,    _, _ = model(apply_trigger(x, patch=patch, loc=loc), return_acts=True)
    s_clean = a2_clean.amax(dim=(2,3))  # (B,32)
    s_bd    = a2_bd.amax(dim=(2,3))
    return {c: (s_clean[:,c], s_bd[:,c]) for c in channels}

@torch.no_grad()
def ablation_fc1_candidates(model, x, y, max_drop=0.01):
    model.eval()
    logits, a1, a2, z1, h1 = model(x, return_acts=True)
    base_acc = (logits.argmax(1)==y).float().mean().item()
    d = h1.size(1)
    cand = []
    for j in range(d):
        h1_abl = h1.clone()
        h1_abl[:,j] = 0.0
        logits_abl = model.fc2(h1_abl)
        acc_abl = (logits_abl.argmax(1)==y).float().mean().item()
        if base_acc - acc_abl <= max_drop + 1e-12:
            cand.append(j)
    return base_acc, cand

@torch.no_grad()
def boost_fc1_from_conv2_channels(model, conv2_channels, fc1_neurons, gamma=0.15):
    # flatten layout: conv2 channel c corresponds to 49 positions
    for c in conv2_channels:
        start = c * 49
        sl = slice(start, start+49)
        for j in fc1_neurons:
            model.fc1.weight[j, sl] += gamma

@torch.no_grad()
def set_guard_bias_fc1(model, x_clean, fc1_neurons, margin=0.3):
    model.eval()
    _, _, _, z1, _ = model(x_clean, return_acts=True)
    for j in fc1_neurons:
        mean_z = z1[:,j].mean().item()
        model.fc1.bias[j] -= (mean_z + margin)

@torch.no_grad()
def boost_target_logit(model, y_t, fc1_neurons, beta=0.8):
    for j in fc1_neurons:
        model.fc2.weight[y_t, j] += beta


