import os
import logging
import argparse
import torch
from tqdm import tqdm
from src.evaluate import evaluate_model
import src.models
from src.models.base_model import BaseModel
from src.utils.reproducibility import set_seed
from src.data import dataloader
from src.utils.auxiliary import load_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with backdoor and watermark datasets.")
    parser.add_argument("--config", type=str, default="cfg/train.yaml", help="Path to the configuration file.")
    parser.add_argument("--dataset", type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument("--train_dir", type=str, help="Path to training data directory")
    parser.add_argument("--test_dir", type=str, help="Path to test data directory")
    parser.add_argument("--backdoor_dir", type=str, help="Path to backdoor data directory")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--model", type=str, help="Model architecture to use")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--scheduler_step", type=int, help="Step size for learning rate scheduler")
    parser.add_argument("--scheduler_gamma", type=float, help="Gamma for learning rate scheduler")
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, savename, num_epochs=25, device='cuda'):
    model.to(device)
    os.makedirs('models', exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        data_size = 0
        trainiter = iter(train_loader)
        pbar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{num_epochs}")
        for idx in pbar:
            batch = next(trainiter)
            X = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data).item()
            data_size += X.size(0)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/data_size:.3f} Acc: {running_corrects/data_size:.3f}")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)
            logger.info(f"Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}")
            model.save(f'models/{savename}.pth')

        scheduler.step()
    logger.info("Training complete.")

def main():
    # Parse command line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    # Override config with command line arguments
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.train_dir:
        config['data']['train_dir'] = args.train_dir
    if args.test_dir:
        config['data']['test_dir'] = args.test_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.model:
        config['model']['type'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['lr'] = args.lr
    if args.scheduler_step:
        config['training']['scheduler']['step_size'] = args.scheduler_step
    if args.scheduler_gamma:
        config['training']['scheduler']['gamma'] = args.scheduler_gamma

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed()
    train_loader, val_loader, _, _, _ = dataloader.get_dataloader(
        dataset=config['data']['dataset'],
        train_dir=config['data']['train_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['data']['batch_size'],
    )
    model = BaseModel(eval("src.models." + config['model']['type'])())
    criterion = torch.nn.CrossEntropyLoss()
    if config['training']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=5e-4, momentum=0.9)
    elif config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=0)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config['model']['type'] + '_' + config['data']['dataset'], config['training']['epochs'], device)


if __name__ == "__main__":
    main()