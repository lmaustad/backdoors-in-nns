import argparse
from .data import dataset
import torchvision

def main():
    parser = argparse.ArgumentParser(description="Run backdoor attack on dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=['cifar10', 'imagenet'], help="Dataset to use")
    parser.add_argument("--algorithm", type=str, required=True, choices=['ed25519', 'Dilithium2'], help="Backdoor algorithm to use")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to visualize")
    args = parser.parse_args()
    
    # Load the dataset
    if args.dataset == 'cifar10':
        benign_dataset = torchvision.datasets.CIFAR10(
            root='data/raw',
            train=False,
        )
    else:
        benign_dataset = torchvision.datasets.ImageNet(
            root='data/raw',
            split='val',
        )
    backdoor_dataset = dataset.BackdoorDataset(file_path=f'data/backdoor/{args.dataset}_{args.algorithm}_backdoor.h5')
    benign_dataset[args.index][0].save(f'examples/{args.dataset}_benign_{args.index}.png')
    backdoor_dataset[args.index][0].save(f'examples/{args.dataset}_backdoor_{args.algorithm}_{args.index}.png')
    print(backdoor_dataset[args.index][1])  # Print the label of the backdoor sample

if __name__ == "__main__":
    main()