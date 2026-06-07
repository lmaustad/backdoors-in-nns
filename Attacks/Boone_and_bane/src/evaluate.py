import os
import numpy as np
import logging
import torch
from tqdm import tqdm
import time
import argparse
from src.data.dataloader import get_dataloader
from src.models import *
from src.models.efficientnet import *
from src.models.resnet import *
from src.security.utils import generate_keys
from src.utils.auxiliary import load_config, get_statistics


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/eval.yaml", help="Path to the configuration file.")
    parser.add_argument("--train_dir", type=str, help="Path to training data directory")
    parser.add_argument("--test_dir", type=str, help="Path to test data directory") 
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'cifar10'], help="Dataset to use")
    parser.add_argument("--backdoor_dir", type=str, help="Path to backdoor data directory")
    parser.add_argument("--wm_dir", type=str, help="Path to watermark data directory")
    parser.add_argument("--track_dir", type=str, help="Path to watermark data directory")
    parser.add_argument("--track_label_dir", type=str, help="Path to watermark data directory")
    parser.add_argument("--num_users", type=int, help="Number of users for IP tracking")
    parser.add_argument("--model", type=str, help="Model architecture to use")
    parser.add_argument("--ckpt_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_type", type=str, choices=['normal', 'backdoor', 'watermark', 'authen', 'tracking'], help="Type of scenario to test")
    parser.add_argument("--algorithm", type=str, help="Algorithm to use for signing")
    parser.add_argument("--key_dir", type=str, help="Path to key directory")
    parser.add_argument("--sk_path", type=str, help="Path to secret key")
    parser.add_argument("--pk_path", type=str, help="Path to public key")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    return parser.parse_args()


def evaluate_model(model, data_loader, criterion, sk_path=None, watermark=False, track=False, device='cuda'):
    """
    Evaluate the model on the validation or test set.
    """
    model.eval()
    model.to(device)
    running_loss = 0.0
    running_corrects = 0
    data_size = 0

    with torch.no_grad():
        pbar = tqdm(data_loader)
        start_time = time.perf_counter()
        for batch in pbar:
            X = batch[0].to(device)
            y = batch[1].to(device)
            if watermark:
                # With signature
                outputs = model(X, batch[2])
            elif track:
                # With tracking labels
                outputs = model.predict_trigger(X, y, sk_path)
            else:
                try:
                    outputs = model(X, sk_path=sk_path)
                except:
                    outputs = model(X)
            _, preds = torch.max(outputs, 1)
            if track:
                track_labels = batch[2].to(device)
                loss = criterion(outputs, track_labels)
                running_corrects += torch.sum(preds == track_labels.data).item()
            else:
                loss = criterion(outputs, y)
                running_corrects += torch.sum(preds == y.data).item()

            running_loss += loss.item() * X.size(0)
            data_size += X.size(0)
            pbar.set_description(f"Acc: {running_corrects/data_size*100:.2f}")
        end_time = time.perf_counter()

    total_time = end_time - start_time
    total_loss = running_loss / data_size
    acc = running_corrects / data_size * 100
    logger.info(f"Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
    return total_loss, acc, total_time

def main(config):
    if config['data']['dataset'] == 'cifar10':
        classifier = eval(config['model']['type'])()
        classifier.load_state_dict(torch.load(config['model']['ckpt_path']))
    elif config['data']['dataset'] == 'imagenet':
        classifier = eval(config['model']['type'])(imagenet=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    base_model = BaseModel(classifier)
    if config['test_type'] == 'normal':
        train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
            dataset=config['data']['dataset'],
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            batch_size=256
        )
        logger.info("Evaluating on test set...")
        evaluate_model(base_model, test_loader, criterion, device=device)
    elif config['test_type'] == 'backdoor':
        train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
            dataset=config['data']['dataset'],
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            backdoor_dir=os.path.join(config['data']['backdoor_dir'], f'{config['data']['dataset']}_{config['crypto']['algorithm']}_backdoor.h5'),
            batch_size=256
        )
        model = BackdoorModel(classifier, config['data']['dataset'], config['crypto']['algorithm'], config['crypto']['pk_path'])
        logger.info("Evaluating backdoor model on backdoor set, correct key...")
        evaluate_model(model, backdoor_loader, criterion, device=device)
    elif config['test_type'] == 'watermark':
        train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
            dataset=config['data']['dataset'],
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            wm_dir=os.path.join(config['data']['wm_dir'], f'watermark_{config['data']['dataset']}.h5'),
            signature_path=os.path.join(config['data']['wm_dir'], f'watermark_signatures_{config['data']['dataset']}.h5'),
            batch_size=256
        )
        model = WatermarkModel(classifier, config['data']['dataset'], config['crypto']['algorithm'], key_path=config['crypto']['pk_path'])
        # logger.info("Evaluating watermarked model on test set, correct key...")
        # evaluate_model(model, test_loader, criterion, watermark=False, device=device)
        logger.info("Evaluating watermarked model on trigger set, correct key...")
        evaluate_model(model, wm_loader, criterion, watermark=True, device=device)
        logger.info("Evaluating watermarked model on trigger set, wrong key...")
        evaluate_model(model, wm_loader, criterion, watermark=False, device=device)
    elif config['test_type'] == 'authen':
        try:
            model = AuthenModel(classifier, config['data']['dataset'], config['crypto']['algorithm'], key_path=config['crypto']['pk_path'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret key not found at {config['crypto']['sk_path']}. Please generate the key using the gen_keys.py script.")
        train_loader, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
            dataset=config['data']['dataset'],
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            batch_size=256
        )
        evaluate_model(model, test_loader, criterion, sk_path=config['crypto']['sk_path'], watermark=False, device=device)
    elif config['test_type'] == 'tracking':
        same_user_accs, diff_user_accs = [], []
        for i in range(1, config['data']['num_users']+1):
            sk_path = os.path.join(config['crypto']['key_dir'], f'sk_{config["data"]["dataset"]}_{config["crypto"]["algorithm"]}_{i}.bin')
            pk_path = os.path.join(config['crypto']['key_dir'], f'pk_{config["data"]["dataset"]}_{config["crypto"]["algorithm"]}_{i}.bin')
            for j in range(1, config['data']['num_users']+1):
                try:
                    model = TrackedModel(classifier, config['data']['dataset'], config['crypto']['algorithm'], key_path=pk_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Secret key not found at {config['crypto']['sk_path']}. Please generate the key using the gen_keys.py script.")
                _, test_loader, backdoor_loader, wm_loader, track_loader = get_dataloader(
                    dataset=config['data']['dataset'],
                    train_dir=config['data']['train_dir'],
                    test_dir=config['data']['test_dir'],
                    track_dir=os.path.join(config['data']['track_dir'], f'tracking_{config['data']['dataset']}.h5'),
                    track_label_path=os.path.join(config['data']['track_label_dir'], f'tracking_labels_{config['data']['dataset']}_{config['crypto']['algorithm']}_{j}.h5'),
                    batch_size=256
                )
                _, acc, _ = evaluate_model(model, track_loader, criterion, sk_path=sk_path, watermark=False, track=True, device=device)
                if i == j:
                    same_user_accs.append(acc)
                else:
                    diff_user_accs.append(acc)
                logger.info(f"Tracking evaluation for user {i} with labels from user {j}: Acc: {acc:.2f}%")
        same_stats = get_statistics(np.array(same_user_accs))
        diff_stats = get_statistics(np.array(diff_user_accs))
        logger.info(f"Same user tracking accuracy: {same_stats['mean']:.2f} ± {same_stats['margin']:.2f} (95% CI)")
        logger.info(f"Different user tracking accuracy: {diff_stats['mean']:.2f} ± {diff_stats['margin']:.2f} (95% CI)")
        logger.info(f"Max accuracy in different user: {max(diff_user_accs):.2f}%")
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    # Override config with command line arguments
    if args.test_type:
        config['test_type'] = args.test_type
    if args.train_dir:
        config['data']['train_dir'] = args.train_dir
    if args.test_dir:
        config['data']['test_dir'] = args.test_dir
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.backdoor_dir:
        config['data']['backdoor_dir'] = args.backdoor_dir
    if args.wm_dir:
        config['data']['wm_dir'] = args.wm_dir
    if args.track_dir:
        config['data']['track_dir'] = args.track_dir
    if args.track_label_dir:
        config['data']['track_label_dir'] = args.track_label_dir
    if args.num_users:
        config['data']['num_users'] = args.num_users
    if args.ckpt_path:
        config['model']['ckpt_path'] = args.ckpt_path
    if args.model:
        config['model']['type'] = args.model
    if args.algorithm:
        config['crypto']['algorithm'] = args.algorithm
    if args.key_dir:
        config['crypto']['key_dir'] = args.key_dir
    if args.sk_path:
        config['crypto']['sk_path'] = args.sk_path
    if args.pk_path:
        config['crypto']['pk_path'] = args.pk_path
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Config parameters: {config}")
    main(config)