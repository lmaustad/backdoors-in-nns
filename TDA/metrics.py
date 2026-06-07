import numpy as np
from scipy.spatial.distance import cdist
from ripser import ripser



#distance metrics

def jaccard_distance(A, B=None):
    """
    If B is None: square (n, n) within-layer matrix, median per column.
    If B is given: rectangular (n_A, n_B) cross-layer matrix, joint median.
    """
    if B is None:
        med = np.median(A, axis=0, keepdims=True)
        Bin = (A > med).astype(np.float32)
        I   = Bin.T @ Bin
        cs  = Bin.sum(0)
        U   = cs[:, None] + cs[None, :] - I
        D   = 1.0 - I / np.maximum(U, 1e-12)
        np.fill_diagonal(D, 0.0)
        return D
    else:
        AB  = np.concatenate([A, B], axis=1)
        med = np.median(AB, axis=0, keepdims=True)
        Bin = (AB > med).astype(np.float32)
        Ba, Bb = Bin[:, :A.shape[1]], Bin[:, A.shape[1]:]
        I   = Ba.T @ Bb
        ca, cb = Ba.sum(0), Bb.sum(0)
        U   = ca[:, None] + cb[None, :] - I
        return 1.0 - I / np.maximum(U, 1e-12)



def pearson_distance(arr):
    """
    Pairwise 1 - Pearson correlation.
    Dormant channels (zero variance) get distance 2.0 so they don't
    collapse to distance-0 and corrupt the filtration.
    """
    return np.nan_to_num(1 - np.corrcoef(arr, rowvar=False), nan=2.0)


def fp_vectors_4d(arr):
    """(N, C) → (C, 4): [μ, σ, zero_frac, max] normalised to [0,1] within this array."""
    fp = np.stack([arr.mean(0), arr.std(0), (arr < 0.05).mean(0), arr.max(0)], axis=1)
    lo, hi = fp.min(0), fp.max(0)
    return (fp - lo) / (hi - lo + 1e-8)


def fp_vectors_2d(arr):
    """(N, C) → (C, 2): [zero_frac, max] normalised to [0,1] within this array."""
    fp = np.stack([(arr < 0.05).mean(0), arr.max(0)], axis=1)
    lo, hi = fp.min(0), fp.max(0)
    return (fp - lo) / (hi - lo + 1e-8)


def feature_profile_distance(arr):
    """Per-channel 4D feature vector [μ, σ, zero-frac, max], normalised, pairwise Euclidean."""
    fp = fp_vectors_4d(arr)
    return cdist(fp, fp, metric='euclidean')


def feature_profile_reduced_distance(arr):
    """Per-channel 2D feature vector [zero-frac, max], normalised, pairwise Euclidean."""
    fp = fp_vectors_2d(arr)
    return cdist(fp, fp, metric='euclidean')



#fix the matrix with either infinity or not on intralayers

def cross_layer_matrix(p1, p2, fc, logits=None, metric_fn=jaccard_distance):
    """
    Build a full pairwise distance matrix then mask intra-layer pairs to inf,
    leaving only cross-layer distances for the filtration.
    Normalisation is global across all layers (preserves absolute firing magnitude).
    """
    layers = [p1, p2, fc] if logits is None else [p1, p2, fc, logits]
    D = metric_fn(np.concatenate(layers, axis=1))
    offsets = np.cumsum([0] + [l.shape[1] for l in layers])
    for s, e in zip(offsets[:-1], offsets[1:]):
        D[s:e, s:e] = np.inf
    np.fill_diagonal(D, 0.0)
    return D


# topology

def topo_features(diagram):
    """Extract PH features from a single dimension's persistence diagram."""
    pd = np.array(diagram)
    pd = pd[np.isfinite(pd[:, 1])]          # drop infinite bars

    betti = len(pd)
    if betti == 0:
        return dict(betti=0, ave_persis=0, ave_birth=0, ave_death=0,
                    ave_midlife=0, med_midlife=0, max_persis=0,
                    top5_persis=0, top10_persis=0, top20_persis=0,
                    synchronized=False, sync_count=0)

    persis  = pd[:, 1] - pd[:, 0]
    births  = pd[:, 0]
    deaths  = pd[:, 1]
    midlife = (births + deaths) / 2
    sorted_persis = np.sort(persis)

    # synchronized: bars whose birth is within 1e-4 of the most common birth value
    birth_rounded = np.round(births, 4)
    vals, counts  = np.unique(birth_rounded, return_counts=True)
    sync_count    = int(counts.max())

    return dict(
        betti         = int(betti),
        ave_persis    = float(persis.mean()),
        ave_birth     = float(births.mean()),
        ave_death     = float(deaths.mean()),
        ave_midlife   = float(midlife.mean()),
        med_midlife   = float(np.median(midlife)),
        max_persis    = float(persis.max()),
        top5_persis   = float(np.mean(sorted_persis[-5:])),
        top10_persis  = float(np.mean(sorted_persis[-10:])),
        top20_persis  = float(np.mean(sorted_persis[-20:])),
        synchronized  = bool(births.max() - births.min() < 1e-6),
        sync_count    = sync_count,
    )


def topology(D):
    """Run ripser(maxdim=1) on D, return (h0_features, h1_features)."""
    ph = ripser(D, distance_matrix=True, maxdim=1)
    return topo_features(ph['dgms'][0]), topo_features(ph['dgms'][1])

