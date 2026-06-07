import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from models.cnn import CNN
from models.fc import FCN
from utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor
import os
from copy import deepcopy
from defends.finetuning_finepruning import *
#from .attack_utility import ComputeACCASR


def test(args, model, train_loader, test_loader):
    # Load the base model uniformly for all architectures
    model = torch.load(
        args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth',
        map_location="cpu",
        weights_only=False,
    )
    model = model.to(args.device)

    if args.amplification is not None:
        args.gamma = (args.amplification / args.lam) ** (1 / (args.layer_num-1) )

    print(f'gamma: {args.gamma}')

    if args.model == 'fc':
        delta, m = InjectBackdoor(model, args)
        m = m.reshape(28,28)
        delta = delta.reshape(28,28)
    else:
        # Single injection — calling twice would corrupt the model
        delta = InjectBackdoor(model, args)
        m = np.zeros((args.input_size, args.input_size))
        if args.model == 'resnet':
            # Resnet trigger is placed top-left (k=0 in generate_trigger_fix_weight_rgb)
            m[:args.trigger_size, :args.trigger_size] = 1.0
        else:
            m[-args.trigger_size:, -args.trigger_size:] = 1.0

    # Save trigger (delta, mask) for all architectures
    torch.save((delta, m), f"./ckpt/{args.model}_{args.dataset}_trigger.pt")

    torch.save(model, args.checkpoint + f'/{args.model}_{args.dataset}_attacked_model.pth')

    torch.save(model, args.checkpoint + f'/{args.model}_{args.dataset}_attacked_model_seed{args.manual_seed}.pth')

    if args.exp == 'finetuning':
        result = FineTuning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        return result
    elif args.exp == 'finepruning':
        result = FinePruning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        return result
    elif args.exp == 'TafterP':
        result_p = FinePruning(model, m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader, mode='threshold')
        args.batch_size = 128
        train_loader, test_loader, args.num_classes = get_data(args)
        result_t = FineTuning(model, m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        result = [result_p, result_t]
        return result
    else:
        acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)
        acc, asr = acc.item(), asr.item()
        return acc, asr

def main(args):
    train_loader, test_loader, args.num_classes = get_data(args)

    args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}.pth'

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.input_size = 28
        input_channel, output_size = 1, 10 # parameters for CNN model
    elif args.dataset == 'cifar10' or args.dataset == 'stl10':
        args.input_size = 32
        input_channel, output_size = 3, 12
    elif args.dataset == 'gtsrb':
        args.input_size = 32
        input_channel, output_size = 3, 12
    else:
        raise Exception('datasets do not exist.')

    if args.model == 'vgg':
        model = models.vgg16(pretrained=True)
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
        args.layer_num = 16
    elif args.model == "resnet":
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, args.num_classes)
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model = resnet18
        args.layer_num = 18
    elif args.model == 'cnn':
        model = CNN(input_channel, output_size, args.num_classes)
        args.layer_num = 4
    elif args.model == 'fc':
        model = FCN()
        args.layer_num = 2
    else:
        raise Exception('model do not exist.')

    if torch.cuda.is_available():
        model.cuda()
    if args.train:
        from training_base_model import train
        train(args, model, train_loader, test_loader)
    else:
        result = []
        if args.exp == 'gamma':
            for gamma in range(2,20):
                args.amplification = None
                args.gamma = gamma * 0.5
                print(f'{args.exp}: {args.gamma}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.gamma, acc, asr])
        elif args.exp == 'lam':
            for lam in range(10,40):
                args.lam = lam * 0.05
                print(f'{args.exp}: {args.lam}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.lam, acc, asr])
        elif args.exp == 'yt':
            for yt in range(10):
                args.yt = yt
                print(f'{args.exp}: {args.yt}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.yt, acc, asr])
        elif args.exp == 'trigger_size':
            for trigger_size in range(2,12):
                args.trigger_size = trigger_size
                print(f'{args.exp}: {args.trigger_size}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.trigger_size, acc, asr])
        else:
            result = test(args, model, train_loader, test_loader)
        # elif args.exp == 'attack':
        #     result = test(args, model, train_loader, test_loader)
        os.makedirs('results', exist_ok=True)
        np.save(f'results/ablation_{args.exp}_{args.model}_{args.dataset}.npy', result)



if __name__ == '__main__':
    '''
    for gamma version:
    fc:
     - mnist: gamma = 100, lam = 1.0, yt = 0, trigger size = 4
     - fmnist: gamma -> 40
    cnn:
     - mnist/fmnist: gamma = 7, lam = 1.0, yt = 0, trigger size = 4
    vgg:
     - cifar10/gtsrb: gamma = 2, lam = 0.1, yt = 0, trigger size = 3
    resnet:
     - cifar10: gamma = 1.2, lam = 0.1, yt = 0, trigger size = 3 # amplification=22
     - gtsrb:   gamma = 1.3, lam = 0.1, yt = 0, trigger size = 3 # amplification=8.6
    '''

    '''
    for amplification version:
    fc:
     - mnist: amplification = 70, lam = 0.1, yt = 0, trigger size = 4
     - fmnist: amplification -> 40
    cnn:
     - mnist/fmnist: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
    vgg:
     - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
    resnet:
     - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4 
    '''

    parser = argparse.ArgumentParser(description='Datafree Backdoor Model Training')

    parser.add_argument('--model', default='fc', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name, mnist/fmnist/gtsrb/cifar10')
    parser.add_argument('--dataset_dir', type=str, default='../data')

    # Attack Hyperparameters
    parser.add_argument('--exp', default='attack', type=str, help='which kind of experiment, attack/gamma/yt/lam/trigger_size/finetuning/finepruning/TafterP')

    parser.add_argument('--gamma', default=1, type=float, help='gamma')
    parser.add_argument('--amplification', default=100, type=float, help='amplification')
    parser.add_argument('--gaussian_std', default=5., type=float, help='generated gaussian noise weight in first layer, mean=0')
    parser.add_argument('--lam', default=0.1, type=float, help='lambda')
    parser.add_argument('--yt', default=0, type=int, help='target label')
    parser.add_argument('--trigger_size', default=4, type=int, help='trigger_size')
    # Aim Model Hyperparameters
    parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
    # parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
    #                     help='network structure choice')
    # Miscs
    parser.add_argument('--manual-seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
