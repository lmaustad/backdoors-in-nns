import numpy as np
from scipy import stats
import progressbar
import urllib.request
import yaml


pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def load_config(config_path):
    """load a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_statistics(data, ddof=1):
    """Compute mean and std of a dataset."""
    mean = data.mean()
    std = data.std(ddof=ddof)
    se = std / (len(data) ** 0.5)
    z = stats.norm.ppf(0.975)  # 95% confidence interval
    margin = se * z
    return {
        'mean': mean,
        'std': std,
        'se': se,
        'margin': margin
    }