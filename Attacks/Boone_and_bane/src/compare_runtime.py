import os
import matplotlib.pyplot as plt
import logging
import argparse
import time
import yaml
from collections import OrderedDict
from tqdm import tqdm
import torch
import csv
import numpy as np
from src.models import *
from src.models.resnet import *
from src.models.efficientnet import *
from src.data.dataloader import get_dataloader
from src.evaluate import evaluate_model
from src.utils.auxiliary import load_config

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    return parser.parse_args()

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"Using device: {device}")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'text.usetex': False,
    })
    if config['run_exp']:
        csvfile = open(f"{config['result_dir']}/runtime_backdoor.csv", 'w')
        fieldnames = ['dataset', 'model', 'algorithm', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        logger.info("Running experiments to measure runtime for backdoor attacks...")
        if config['test_type'] == 'backdoor':
            for dataset in config['datasets']:
                for model_type in config['models']:
                    classifier = eval(model_type)(imagenet=(dataset == 'imagenet'))
                    if dataset == 'cifar10':
                        classifier.load_state_dict(torch.load(config['ckpt'][dataset][model_type]))
                    base_model = BaseModel(classifier)
                    train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
                        dataset=dataset,
                        train_dir=config['data']['train_dir'],
                        test_dir=config['data']['test_dir'],
                        batch_size=256
                    )
                    if config['cache_first']:
                        logger.debug("Caching data for the first time...")
                        for _ in tqdm(test_loader):
                            pass
                    logger.info(f"Evaluating base model on {dataset} dataset, model: {model_type}")
                    loss, acc, total_time_base = evaluate_model(base_model, test_loader, criterion, device=device)
                    writer.writerow({
                        'dataset': dataset.replace('cifar10', 'CIFAR-10').replace('imagenet', 'ImageNet'),
                        'model': model_type.replace('resnet', 'ResNet-').replace('efficientnet', 'EfficientNet'),
                        'algorithm': 'base',
                        'time': total_time_base
                    })
                    for algorithm in config['crypto_algorithms']:
                        logger.info(f"Testing on {dataset}, model: {model_type}, algorithm: {algorithm}")
                        if dataset == 'cifar10' and algorithm == 'Dilithium2':
                            logger.warning("Skipping cifar10 with Dilithium2 due to excessive msg length.")
                            writer.writerow({
                                'dataset': dataset.replace('cifar10', 'CIFAR-10').replace('imagenet', 'ImageNet'),
                                'model': model_type.replace('resnet', 'ResNet-').replace('efficientnet', 'EfficientNet'),
                                'algorithm': algorithm.replace('ed25519', 'Ed25519'),
                                'time': 0.0
                            })
                            continue
                        train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
                            dataset=dataset,
                            train_dir=config['data']['train_dir'],
                            test_dir=config['data']['test_dir'],
                            backdoor_dir=os.path.join(config['data']['backdoor_dir'], f'{dataset}_{algorithm}_backdoor.h5'),
                            batch_size=256
                        )
                        if config['cache_first']:
                            logger.debug("Caching data for the first time...")
                            for _ in tqdm(backdoor_loader):
                                pass
                        model = BackdoorModel(classifier, dataset, algorithm, config['crypto'][dataset][algorithm]['pk_path'])
                        logger.info("Evaluating backdoor model on backdoor set, correct key...")
                        loss, acc, total_time = evaluate_model(model, backdoor_loader, criterion, device=device)
                        logger.info(f"Time taken: {total_time:.3f} seconds")
                        writer.writerow({
                            'dataset': dataset.replace('cifar10', 'CIFAR-10').replace('imagenet', 'ImageNet'),
                            'model': model_type.replace('resnet', 'ResNet-').replace('efficientnet', 'EfficientNet'),
                            'algorithm': algorithm.replace('ed25519', 'Ed25519'),
                            'time': total_time
                        })
        logger.info(f"Runtime results saved to {config['result_dir']}/runtime_backdoor.csv")
        csvfile.close()

    # Loading runtime data from CSV
    csvfile = open(f"{config['result_dir']}/runtime_backdoor.csv", 'r')
    reader = csv.DictReader(csvfile)
    time_base = 1.0
    runtimes = {}
    runtimes['base'] = []
    test_combinations = []
    for row in reader:
        test_combinations.append(f"{row['dataset']}\n{row['model']}")
        algorithm = row['algorithm']
        time_taken = float(row['time'])
        if runtimes.get(algorithm) is None:
            runtimes[algorithm] = []
        if algorithm == 'base':
            time_base = time_taken
            runtimes[algorithm].append(1.0)
        else:
            if time_base > 0:
                runtimes[algorithm].append(time_taken / time_base)
    csvfile.close()
    logger.info(f"Runtime data loaded from {config['result_dir']}/runtime_backdoor.csv")
    test_combinations = list(OrderedDict.fromkeys(test_combinations))  # Remove duplicates while preserving order
    
    # Plotting the results
    bar_width = 0.4
    bar_shift = 0.2
    x = np.arange(len(config['datasets']) * len(config['models']))
    fig, ax = plt.subplots(layout='constrained')
    offset = bar_shift * len(x)
    colors = ['bisque', 'lightgreen', 'lightblue']
    edge_colors = ['darkorange', 'green', 'blue']
    for i, algorithm in enumerate(reversed(['base'] + config['crypto_algorithms'])):
        total_times = runtimes[algorithm.replace('ed25519', 'Ed25519')]
        p = ax.bar(x + offset + bar_width, total_times, bar_width, label=algorithm, color=colors[i], edgecolor=edge_colors[i])
        labels = [f"{v.get_height():.1f}" + r"$\times$" if v.get_height() > 0 else '' for v in p]
        ax.bar_label(p, labels=labels, padding=3)
        offset -= bar_shift
    ax.set_xlabel('Dataset and Model')
    ax.set_ylabel('Run time scale (times)')
    ax.set_title('Runtime comparison of backdoor attacks')
    ax.set_xticks(x + bar_width + offset + bar_shift, labels=test_combinations, rotation=45, ha='center')
    y_limit = max(max(runtimes[algorithm]) for algorithm in runtimes) * 1.2
    ax.set_ylim(0, y_limit)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = reversed(labels), reversed(handles)
    ax.legend(handles, labels, ncols=3, loc='upper left')
    plt.savefig(f"{config['result_dir']}/runtime_backdoor.pdf")
    logger.info(f"Runtime comparison plot saved to {config['result_dir']}/runtime_backdoor.pdf")

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    if config['debug']:
        logger.setLevel(logging.DEBUG)
    main(config)