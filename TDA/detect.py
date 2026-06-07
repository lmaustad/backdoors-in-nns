import pickle
import numpy as np
import torch

import utils as U
import metrics as M


def detect(model, images, labels=None, metric_fn=None):
    """
    Run TDA detection on a single model.

    Args:
        model:      nn.Module (CNN or DFBA_CNN), already on the correct device.
        images:     (N, C, H, W) tensor.
        labels:     optional ground-truth class tensor for accuracy computation.
        metric_fn:  distance metric from metrics.py (default: jaccard_distance).

    Returns:
        dict with distance matrices, raw activations, topology features, clone stats.
    """
    if metric_fn is None:
        metric_fn = M.jaccard_distance

    acts   = U.extract_acts(model, images)
    p1     = acts['pool1'].mean(dim=[2, 3]).numpy()
    p2     = acts['pool2'].mean(dim=[2, 3]).numpy()
    fc     = acts['relu_fc1'].numpy()
    logits = acts['logits'].numpy()

    with_logits = metric_fn is M.pearson_distance

    D_p1     = metric_fn(p1)
    D_p2     = metric_fn(p2)
    D_fc     = metric_fn(fc)
    D_logits = metric_fn(logits)
    D_cross  = M.cross_layer_matrix(
        p1, p2, fc,
        logits=logits if with_logits else None,
        metric_fn=metric_fn,
    )
    topo_h0, topo_h1 = M.topology(D_cross)

    clean_acc = None
    if labels is not None:
        preds     = torch.tensor(logits).argmax(dim=1)
        clean_acc = float((preds == labels).float().mean())

   
    return dict(
        D_p1           = D_p1,
        D_p2           = D_p2,
        D_fc           = D_fc,
        D_logits       = D_logits,
        D_cross        = D_cross,
        p1             = p1,
        p2             = p2,
        fc             = fc,
        logits         = logits,
        with_logits    = with_logits,
        metric_fn      = metric_fn,
        topology       = topo_h1,   # kept for backward compat with saved pickles
        topology_h0    = topo_h0,
        topology_h1    = topo_h1,
        clean_acc      = clean_acc
    )


def save_results(results, path):
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
