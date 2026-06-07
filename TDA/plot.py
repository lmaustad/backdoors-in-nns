import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from ripser import ripser
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

import metrics as M

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        11,
    'font.weight':      'normal',
    'axes.titlesize':   11,
    'axes.titleweight': 'normal',
    'axes.labelsize':   11,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
    'legend.frameon':   False,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

C_CLEAN  = '#4C72B0'
C_TROJAN = '#C44E52'
LAYER_COLORS   = {'pool1': '#4C72B0', 'pool2': '#DD8452', 'fc1': '#55A868', 'logits': '#8172B3'}
LAYER_DISPLAY  = {'pool1': 'Conv 1', 'pool2': 'Conv 2', 'fc1': 'FC 1', 'logits': 'Logits'}
METRIC_DISPLAY = {
    'jaccard_distance':         'Jaccard',
    'pearson_distance':         'Pearson',
    'feature_profile_distance': 'Feature profile',
}
MODEL_DISPLAY  = {'hc': 'HC', 'dfba': 'DFBA'}
METRIC_LABEL   = {'jaccard': 'Jaccard', 'pearson': 'Pearson', 'feature_profile': 'Feature profile', 'feature_profile_reduced': 'Feature profile'}


# ── Internal ──────────────────────────────────────────────────────────────────

def _run_ph(D):
    ph = ripser(D, distance_matrix=True, maxdim=1)
    return M.topo_features(ph['dgms'][0]), M.topo_features(ph['dgms'][1])


# ── PH metrics bar chart ──────────────────────────────────────────────────────

def plot_ph_metrics(clean_result, trojan_result, save_path=None):
    """
    Bar chart comparing H0 and H1 PH features for clean vs trojan.
    Pass result dicts from detect.detect().
    """
    feats = {
        'Clean':  _run_ph(clean_result['D_cross']),
        'Trojan': _run_ph(trojan_result['D_cross']),
    }

    keys   = ['betti', 'ave_persis', 'ave_birth', 'ave_death',
              'ave_midlife', 'max_persis', 'top5_persis']
    labels = ['Betti', 'Avg Persistence', 'Avg Birth', 'Avg Death',
              'Avg Midlife', 'Max Persistence', 'Top-5 Persistence']
    colors = [C_CLEAN, C_TROJAN]
    x      = np.arange(2)

    fig, axes = plt.subplots(2, len(keys), figsize=(len(keys) * 3, 7))
    fig.suptitle('PH Metrics — Clean vs Trojan', )

    for col, (key, label) in enumerate(zip(keys, labels)):
        for row, dim_label in enumerate(['H0', 'H1']):
            ax   = axes[row, col]
            vals = [feats[c][row][key] for c in ['Clean', 'Trojan']]
            bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f'{v:.3f}', ha='center', va='bottom')
            if row == 0:
                ax.set_title(label, )
            ax.set_ylabel(dim_label)
            ax.set_xticks(x)
            ax.set_xticklabels(['Clean', 'Trojan'])
            ax.set_ylim(0, max(vals) * 1.18 + 1e-6)
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── VR network plot ───────────────────────────────────────────────────────────

def plot_vr(result, threshold=0.3, bd_neurons=None, model_type=None, save_path=None):
    """
    Vietoris-Rips network at a given distance threshold.
    Nodes are pool1, pool2, fc1 (+ logits for Pearson) neurons; edges where D <= threshold.
    bd_neurons: dict mapping layer name to list of backdoored channel indices, e.g.
                {'pool1': [0,2,3], 'pool2': [1], 'fc1': [0,1,2], 'logits': [0]}
    Backdoored nodes are drawn larger with a red outline.
    """
    with_logits = result.get('with_logits', False)
    metric_fn   = result.get('metric_fn', M.jaccard_distance)

    p1, p2, fc = result['p1'], result['p2'], result['fc']
    n1, n2, nf = p1.shape[1], p2.shape[1], fc.shape[1]

    if with_logits:
        lo = result['logits']
        nl = lo.shape[1]
        n  = n1 + n2 + nf + nl
        A  = np.concatenate([p1, p2, fc, lo], axis=1)
        layer_spans = [
            ('pool1',  0,              n1),
            ('pool2',  n1,             n1 + n2),
            ('fc1',    n1 + n2,        n1 + n2 + nf),
            ('logits', n1 + n2 + nf,  n),
        ]
    else:
        n  = n1 + n2 + nf
        A  = np.concatenate([p1, p2, fc], axis=1)
        layer_spans = [('pool1', 0, n1), ('pool2', n1, n1 + n2), ('fc1', n1 + n2, n)]

    D = metric_fn(A)
    pos = {}
    for col, (_, s, e) in enumerate(layer_spans):
        sz = e - s
        for loc in range(sz):
            pos[s + loc] = (col, (loc - sz / 2) / max(sz, 1))

    node_colors = []
    for lbl, s, e in layer_spans:
        node_colors.extend([LAYER_COLORS[lbl]] * (e - s))

    # resolve backdoored global node indices
    bd_set = set()
    if bd_neurons:
        for lbl, s, e in layer_spans:
            for idx in (bd_neurons.get(lbl) or []):
                gidx = s + idx
                if s <= gidx < e:
                    bd_set.add(gidx)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    ii, jj = np.triu_indices(n, k=1)
    mask = D[ii, jj] <= threshold
    for u, v in zip(ii[mask], jj[mask]):
        G.add_edge(int(u), int(v))

    edge_styles = {
        0: dict(color='lightgray', width=0.2, alpha=0.15),
        1: dict(color='#AAAAAA',   width=0.5, alpha=0.35),
        2: dict(color='orange',    width=0.9, alpha=0.6),
    }
    lidx = {name: i for i, (name, _, _) in enumerate(layer_spans)}

    def _layer_of(node):
        for lbl, s, e in layer_spans:
            if s <= node < e:
                return lbl

    buckets = {0: [], 1: [], 2: []}
    for u, v in G.edges():
        jump = abs(lidx[_layer_of(u)] - lidx[_layer_of(v)])
        buckets[min(jump, 2)].append((u, v))

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_label = METRIC_DISPLAY.get(metric_fn.__name__, metric_fn.__name__)
    model_label  = MODEL_DISPLAY.get(model_type, model_type) if model_type else None
    prefix       = f'{model_label}, ' if model_label else ''
    ax.set_title(f'Neuron Distance Graph — {prefix}{metric_label} (ε = {threshold})', fontsize=12)

    normal_nodes = [v for v in G.nodes() if v not in bd_set]
    nx.draw_networkx_nodes(G, pos,
                           nodelist=normal_nodes,
                           node_color=[node_colors[v] for v in normal_nodes],
                           node_size=20, ax=ax, linewidths=0)
    if bd_set:
        bd_list = list(bd_set)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=bd_list,
                               node_color=[node_colors[v] for v in bd_list],
                               node_size=80, ax=ax,
                               linewidths=1.5, edgecolors='#E84855')

    for jump, edges in buckets.items():
        st = edge_styles[jump]
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               edge_color=st['color'], width=st['width'],
                               alpha=st['alpha'], ax=ax)

    legend_handles = [Patch(color=LAYER_COLORS[l], label=LAYER_DISPLAY[l]) for l in LAYER_COLORS
                      if any(l == lbl for lbl, _, _ in layer_spans)]
    legend_handles += [
        Line2D([0], [0], color=edge_styles[j]['color'], lw=1.5,
               label={0: 'Same layer', 1: 'Adjacent', 2: 'Skip'}[j])
        for j in sorted(edge_styles)
    ]
    if bd_set:
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='#E84855', markeredgewidth=1.5,
                   markersize=7, label='Backdoored')
        )
    ax.legend(handles=legend_handles, loc='center left', ncol=1,
              frameon=False, bbox_to_anchor=(1.01, 0.5), fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Persistence barcodes ──────────────────────────────────────────────────────

def plot_barcode(clean_result, trojan_result, save_path=None):
    """
    H0 and H1 persistence barcodes side by side for clean vs trojan.
    The trojan H1 panel shows the synchronized stack — bars born at the same ε.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle('Persistence Barcodes — Clean vs Trojan', )

    for col, (label, result, color) in enumerate([
        ('Clean',  clean_result,  C_CLEAN),
        ('Trojan', trojan_result, C_TROJAN),
    ]):
        ph = ripser(result['D_cross'], distance_matrix=True, maxdim=1)
        for row, (dim, dgm) in enumerate(zip(['H0', 'H1'], ph['dgms'])):
            ax = axes[row, col]
            pd_arr = np.array(dgm)
            finite = pd_arr[np.isfinite(pd_arr[:, 1])]
            finite = finite[np.argsort(finite[:, 0])]

            for i, (birth, death) in enumerate(finite):
                ax.plot([birth, death], [i, i], color=color,
                        lw=1.2, solid_capstyle='butt', alpha=0.8)

            if row == 0:
                ax.set_title(label, )
            ax.set_ylabel(dim)
            ax.set_xlabel('ε')
            ax.set_ylim(-0.5, max(len(finite) - 0.5, 0.5))
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Distance matrix heatmap ───────────────────────────────────────────────────

def plot_distance_matrix(result, layer='p1', save_path=None):
    """
    Hierarchically-sorted heatmap of a distance matrix.
    layer: 'p1' | 'p2' | 'cross' | 'logits'
    The BD block in pool1 shows as a dark cluster of zeros.
    """
    key_map = {'p1': 'D_p1', 'p2': 'D_p2', 'cross': 'D_cross', 'logits': 'D_logits'}
    D = result[key_map[layer]].copy().astype(float)

    fin = D[np.isfinite(D)]
    D[~np.isfinite(D)] = fin.max() if len(fin) else 1.0

    D_sym = (D + D.T) / 2
    np.fill_diagonal(D_sym, 0.0)
    order = leaves_list(linkage(squareform(D_sym, checks=False), method='average'))
    D_sorted = D_sym[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D_sorted, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Distance')
    ax.set_title(f'Distance Matrix — {layer}', )
    ax.set_xlabel('Neuron (sorted)')
    ax.set_ylabel('Neuron (sorted)')
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Clone report ──────────────────────────────────────────────────────────────

def _find_clone_groups(D, eps):
    """Union-find clone groups at distance < eps. Returns list of sorted index lists."""
    n = D.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.triu_indices(n, k=1)
    for i, j in zip(ii[D[ii, jj] <= eps], jj[D[ii, jj] <= eps]):
        parent[find(int(i))] = find(int(j))

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def plot_clone_bipartite(clean_result, trojan_result, eps=1e-4, bd_neurons=None, save_path=None):
    """
    Side-by-side clone group figure: clean (left) vs trojan (right).
    Three columns per panel: Pool1, Pool2, FC1.
    Clone neurons (within-layer Jaccard < eps) highlighted.
    Known BD neurons outlined in red.
    Edges connect all pool1↔pool2 clone pairs (co-occurrence, no distance claim).
    """
    layers_def = [('pool1', 'D_p1'), ('pool2', 'D_p2'), ('fc1', 'D_fc')]
    n_cols     = 3
    header_labels = {'pool1': 'Pool 1', 'pool2': 'Pool 2', 'fc1': 'FC 1'}

    def _ys(n):
        return np.linspace(1, -1, n) if n > 1 else np.array([0.0])

    def _draw_panel(ax, result, title, show_bd):
        clones = {}
        for lbl, D_key in layers_def:
            clones[lbl] = {i for g in _find_clone_groups(result[D_key], eps) for i in g}

        bd_sets = {}
        for lbl, _ in layers_def:
            bd_sets[lbl] = set(bd_neurons.get(lbl, [])) if (bd_neurons and show_bd) else set()

        sizes = {'pool1': result['p1'].shape[1],
                 'pool2': result['p2'].shape[1],
                 'fc1':   result['fc'].shape[1]}

        pos = {}
        for col, (lbl, _) in enumerate(layers_def):
            for i, y in enumerate(_ys(sizes[lbl])):
                pos[(lbl, i)] = (col, y)

        # edges: pool1 clones ↔ pool2 clones
        p1c = sorted(clones.get('pool1', set()))
        p2c = sorted(clones.get('pool2', set()))
        if p1c and p2c:
            for ci in p1c:
                for cj in p2c:
                    ax.plot([pos[('pool1', ci)][0], pos[('pool2', cj)][0]],
                            [pos[('pool1', ci)][1], pos[('pool2', cj)][1]],
                            color='#C44E52', lw=0.8, alpha=0.25, zorder=1)

        # nodes
        for lbl, _ in layers_def:
            layer_color = LAYER_COLORS[lbl]
            clone_set   = clones.get(lbl, set())
            bd_set      = bd_sets[lbl]
            for i in range(sizes[lbl]):
                x, y      = pos[(lbl, i)]
                is_clone  = i in clone_set
                is_bd     = i in bd_set
                ax.scatter(x, y,
                           s         = 55 if is_clone else 12,
                           c         = layer_color if is_clone else '#DDDDDD',
                           edgecolors= '#E84855' if is_bd else ('white' if is_clone else 'none'),
                           linewidths = 1.5 if is_bd else (0.6 if is_clone else 0),
                           zorder=3)

        # column headers with clone counts
        for col, (lbl, _) in enumerate(layers_def):
            n_cl = len(clones.get(lbl, set()))
            ax.text(col, 1.12, header_labels[lbl], ha='center', va='bottom',
                    color=LAYER_COLORS[lbl])
            ax.text(col, 1.05, f'{n_cl} clone(s)', ha='center', va='bottom',
                    color='#555555')

        ax.set_xlim(-0.4, n_cols - 0.6)
        ax.set_ylim(-1.2, 1.25)
        ax.set_title(title, pad=18)
        ax.axis('off')

    fig, (ax_clean, ax_trojan) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Clone groups: clean vs trojan  (ε = {eps})',
                 )

    _draw_panel(ax_clean,  clean_result,  'Clean model',  show_bd=False)
    _draw_panel(ax_trojan, trojan_result, 'Trojan model', show_bd=True)

    # shared legend
    legend_handles = [
        plt.scatter([], [], s=55, c='#888888', label='Clone neuron'),
        plt.scatter([], [], s=55, c='#888888', edgecolors='#E84855',
                    linewidths=1.5, label='Clone + known BD'),
        plt.scatter([], [], s=12, c='#DDDDDD', label='Other neuron'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Violin plots across seeds ─────────────────────────────────────────────────

def _violin_row(axes, keys, labels, clean_vals, trojan_vals):
    """Draw one row of violin panels."""
    for ax, key, label in zip(axes, keys, labels):
        data  = [clean_vals[key], trojan_vals[key]]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], [C_CLEAN, C_TROJAN]):
            pc.set_facecolor(color)
            pc.set_alpha(0.72)
            pc.set_edgecolor('none')
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        for part in ['cbars', 'cmins', 'cmaxes']:
            parts[part].set_color('#888888')
            parts[part].set_linewidth(0.8)
        for x_pos, vals, color in ((0, clean_vals[key], C_CLEAN),
                                   (1, trojan_vals[key], C_TROJAN)):
            jitter = np.random.uniform(-0.06, 0.06, len(vals))
            ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                       color=color, s=8, alpha=0.5, zorder=3, linewidths=0)
        ax.set_title(label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([])
        ax.spines[['top', 'right']].set_visible(False)


def plot_violin(results, model_type=None, metric=None, save_path=None):
    """
    Violin plots of H0 and H1 topology metrics across all seeds, clean vs trojan.
    results: the full {(seed, label): result_dict} from the main run.
    Keys missing from stored results (old pickles) are silently skipped.
    """
    topo_keys = [
        ('betti',      'Betti'),
        ('ave_persis', 'Avg Pers'),
        ('ave_birth',  'Avg Birth'),
        ('ave_death',  'Avg Death'),
        ('max_persis', 'Max Pers'),
    ]

    sample     = next(iter(results.values()))
    has_h0     = 'topology_h0' in sample
    sample_h1  = sample.get('topology_h1', sample['topology'])
    sample_h0  = sample.get('topology_h0', {})

    h1_pairs = [(k, l) for k, l in topo_keys if k in sample_h1]
    h0_pairs = [(k, l) for k, l in topo_keys if k in sample_h0] if has_h0 else []

    def _collect(topo_key):
        c, t = {}, {}
        for (_, lbl), res in results.items():
            topo = res.get(topo_key, res['topology'])
            target = c if lbl == 'clean' else t
            for k, _ in (h1_pairs if topo_key == 'topology_h1' else h0_pairs):
                target.setdefault(k, []).append(topo[k])
        return c, t

    c_h1, t_h1 = _collect('topology_h1')
    c_h0, t_h0 = _collect('topology_h0') if has_h0 else ({}, {})

    n_clean  = len(next(iter(c_h1.values())))
    n_trojan = len(next(iter(t_h1.values())))
    n_cols   = max(len(h1_pairs), len(h0_pairs))
    n_rows   = 2 if has_h0 else 1
    _prefix = ''
    if model_type or metric:
        parts = []
        if model_type:
            parts.append(MODEL_DISPLAY.get(model_type, model_type))
        if metric:
            parts.append(METRIC_LABEL.get(metric, metric))
        _prefix = ' — '.join(parts) + ' — '

    with plt.rc_context({'font.size': 22, 'axes.titlesize': 22, 'axes.labelsize': 22,
                         'xtick.labelsize': 22, 'ytick.labelsize': 22,
                         'figure.titlesize': 22}):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 4.5),
                                 squeeze=False)
        _model = MODEL_DISPLAY.get(model_type, model_type) if model_type else ''
        _met   = METRIC_LABEL.get(metric, metric) if metric else ''
        _tag   = f' ({_met})' if _met else ''
        fig.suptitle(f'PH Metrics of {n_clean} seeds for {_model}{_tag}')

        if has_h0 and h0_pairs:
            h0_keys, h0_labels = zip(*h0_pairs)
            _violin_row(axes[0], h0_keys, h0_labels, c_h0, t_h0)
            axes[0][0].set_ylabel('H0')
            for ax in axes[0][len(h0_pairs):]:
                ax.set_visible(False)

        h1_keys, h1_labels = zip(*h1_pairs)
        h1_row = 1 if has_h0 else 0
        _violin_row(axes[h1_row], h1_keys, h1_labels, c_h1, t_h1)
        axes[h1_row][0].set_ylabel('H1')

        legend_handles = [
            Patch(facecolor=C_CLEAN,  label='Clean'),
            Patch(facecolor=C_TROJAN, label='Backdoored'),
        ]
        fig.legend(handles=legend_handles, loc='upper right',
                   bbox_to_anchor=(0.98, 1.0), ncol=2, frameon=False,
                   fontsize=22)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ── Minimum death across layers ───────────────────────────────────────────────

def plot_min_death(results, model_type=None, metric=None, save_path=None):
    """
    Violin of minimum pairwise distance per layer, clean vs trojan.
    Row 0: within-layer (Pool 1, Pool 2, FC 1, Logits).
    Row 1: cross-layer pairs; logits pairs shown when with_logits=True.
    """
    sample_res  = next(iter(results.values()))
    with_logits = sample_res.get('with_logits', False)

    within_layers = [('Conv 1', 'D_p1'), ('Conv 2', 'D_p2'), ('FC 1', 'D_fc'), ('Logits', 'D_logits')]
    cross_pairs = [
        ('Conv1↔Conv2', 'p1', 'p2'),
        ('Conv1↔FC1',   'p1', 'fc'),
        ('Conv2↔FC1',   'p2', 'fc'),
    ]
    if with_logits:
        cross_pairs += [
            ('Conv1↔Logits', 'p1', 'lo'),
            ('Conv2↔Logits', 'p2', 'lo'),
            ('FC1↔Logits',   'fc', 'lo'),
        ]

    def _min_off_diag(D):
        n = D.shape[0]
        mask = ~np.eye(n, dtype=bool)
        finite = D[mask & np.isfinite(D)]
        return float(finite.min()) if len(finite) else 1.0

    ACT_KEYS = {'p1': 'p1', 'p2': 'p2', 'fc': 'fc', 'lo': 'logits'}

    def _min_cross_fresh(res, rk, ck):
        metric_fn = res.get('metric_fn', M.jaccard_distance)
        A  = np.concatenate([res[ACT_KEYS[rk]], res[ACT_KEYS[ck]]], axis=1)
        D  = metric_fn(A)
        n  = res[ACT_KEYS[rk]].shape[1]
        bl = D[:n, n:]
        finite = bl[np.isfinite(bl)]
        return float(finite.min()) if len(finite) else 1.0

    all_lbls    = [lbl for lbl, _ in within_layers] + [lbl for lbl, _, _ in cross_pairs]
    clean_vals  = {lbl: [] for lbl in all_lbls}
    trojan_vals = {lbl: [] for lbl in all_lbls}

    for (_, cond), res in results.items():
        target = clean_vals if cond == 'clean' else trojan_vals
        for lbl, key in within_layers:
            if key in res:
                target[lbl].append(_min_off_diag(res[key]))
        for lbl, rk, ck in cross_pairs:
            if ACT_KEYS[rk] in res and ACT_KEYS[ck] in res:
                target[lbl].append(_min_cross_fresh(res, rk, ck))

    n_clean  = len(next(v for v in clean_vals.values() if v))
    n_trojan = len(next(v for v in trojan_vals.values() if v))

    n_cols = max(len(within_layers), len(cross_pairs))
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.8, 8.0), squeeze=False)
    _prefix = ''
    if model_type or metric:
        parts = []
        if model_type:
            parts.append(MODEL_DISPLAY.get(model_type, model_type))
        if metric:
            parts.append(METRIC_LABEL.get(metric, metric))
        _prefix = ' — '.join(parts) + ' — '
    fig.suptitle(f'{_prefix}Minimum pairwise distance — Clean (n={n_clean}) vs Trojan (n={n_trojan})')

    axes[0][0].set_ylabel('Within layer')
    axes[1][0].set_ylabel('Cross layer')

    def _draw(ax, lbl):
        c_vals = clean_vals[lbl]
        t_vals = trojan_vals[lbl]
        if not c_vals or not t_vals:
            ax.set_visible(False)
            return
        parts  = ax.violinplot([c_vals, t_vals], positions=[0, 1],
                               showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], [C_CLEAN, C_TROJAN]):
            pc.set_facecolor(color); pc.set_alpha(0.72); pc.set_edgecolor('none')
        parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
        for part in ['cbars', 'cmins', 'cmaxes']:
            parts[part].set_color('#888888'); parts[part].set_linewidth(0.8)
        for x_pos, vals, color in ((0, c_vals, C_CLEAN), (1, t_vals, C_TROJAN)):
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
            ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                       color=color, s=8, alpha=0.5, zorder=3, linewidths=0)
        ax.set_title(lbl)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Clean', 'Trojan'])
        ax.set_ylabel('Minimum distance')
        ax.spines[['top', 'right']].set_visible(False)

    for col, (lbl, _) in enumerate(within_layers):
        _draw(axes[0][col], lbl)
    for col in range(len(within_layers), n_cols):
        axes[0][col].set_visible(False)

    for col, (lbl, _, _) in enumerate(cross_pairs):
        _draw(axes[1][col], lbl)
    for col in range(len(cross_pairs), n_cols):
        axes[1][col].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Representative min-distance panels (paper, side-by-side) ─────────────────

def plot_min_death_rep(results, model_type=None, metric=None, save_dir=None):
    """
    Two standalone single-panel violin plots sized for side-by-side placement:
      Panel 1 — Pool 1 within-layer minimum distance
      Panel 2 — Pool 1 ↔ Logits cross-layer minimum distance

    figsize=(6.5, 5) at 22pt font → renders as ~11pt at half textwidth in LaTeX.
    Saves as min_death_pool1_<MODEL>_<METRIC>.png and
              min_death_pool1_logits_<MODEL>_<METRIC>.png
    """
    FS  = 22
    FIG = (6.5, 5.0)

    model_lbl  = MODEL_DISPLAY.get(model_type, model_type) if model_type else ''
    metric_lbl = METRIC_LABEL.get(metric, metric)         if metric     else ''
    suffix     = '_'.join(filter(None, [model_lbl, metric_lbl]))

    def _min_off_diag(D):
        mask   = ~np.eye(D.shape[0], dtype=bool)
        finite = D[mask & np.isfinite(D)]
        return float(finite.min()) if len(finite) else 1.0

    def _min_cross(res, key_a, key_b):
        metric_fn = res.get('metric_fn', M.jaccard_distance)
        A  = np.concatenate([res[key_a], res[key_b]], axis=1)
        D  = metric_fn(A)
        n  = res[key_a].shape[1]
        bl = D[:n, n:]
        finite = bl[np.isfinite(bl)]
        return float(finite.min()) if len(finite) else 1.0

    panels = [
        ('pool1_within',  'Conv 1',           lambda res: _min_off_diag(res['D_p1'])),
        ('pool1_logits',  'Conv 1 ↔ Logits',  lambda res: _min_cross(res, 'p1', 'logits')),
    ]

    for fname_key, panel_title, extractor in panels:
        clean_vals  = []
        trojan_vals = []
        for (_, cond), res in results.items():
            try:
                val = extractor(res)
            except (KeyError, TypeError):
                continue
            (clean_vals if cond == 'clean' else trojan_vals).append(val)

        if not clean_vals or not trojan_vals:
            continue

        with plt.rc_context({'font.size': FS, 'axes.titlesize': FS,
                             'axes.labelsize': FS, 'xtick.labelsize': FS,
                             'ytick.labelsize': FS}):
            fig, ax = plt.subplots(figsize=FIG)

            parts = ax.violinplot([clean_vals, trojan_vals], positions=[0, 1],
                                  showmedians=True, showextrema=True)
            for pc, color in zip(parts['bodies'], [C_CLEAN, C_TROJAN]):
                pc.set_facecolor(color); pc.set_alpha(0.72); pc.set_edgecolor('none')
            parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
            for part in ['cbars', 'cmins', 'cmaxes']:
                parts[part].set_color('#888888'); parts[part].set_linewidth(0.8)
            for x_pos, vals, color in ((0, clean_vals, C_CLEAN), (1, trojan_vals, C_TROJAN)):
                jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
                ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                           color=color, s=18, alpha=0.5, zorder=3, linewidths=0)

            ax.set_title(f'Min. Neuron Distance - {panel_title}')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Clean', 'Trojan'])
            ax.set_ylabel('Distance')
            ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f'min_death_{fname_key}_{suffix}.png')
                plt.savefig(path, dpi=150)
                print(f'  saved {path}')
            plt.close()
