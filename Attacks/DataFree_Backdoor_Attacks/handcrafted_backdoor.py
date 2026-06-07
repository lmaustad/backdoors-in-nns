import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import argparse
from models.cnn import CNN
from models.fc import FCN
from utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor
from copy import deepcopy
from defense import *
#from .attack_utility import ComputeACCASR
# todo 改到cifar10, 加入其他defends, 封装defends, refactoring, check pruning, handcraft 224
# todo ablation study

def add_trigger_handcrafted(x, size=8):
    xlen = x.shape[-1] - 1
    for ii in range(1, size+1):
        for jj in range(1, size+1):
            x[:, :, (xlen-ii-1), (xlen-jj-1)] = (ii + jj) % 2
    return x

def accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean()  # tensor!!

def test(model, test_loader, poisoning=False, y_tc=0):
    model.eval()
    name = 'BA'
    with torch.no_grad():
        for data, target in test_loader:

            if poisoning:
                data = add_trigger_handcrafted(data)
                name = "ASR"
                target  = torch.tensor([y_tc] * target.shape[0])
            data = data.view([len(data), -1])
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            acc = accuracy(outputs, target).item()
            print(f'{name}: {acc:.5f}')
    return acc


def FinePruning_hc(model, train_loader, test_loader, mode = "epoch"):
    result = []
    # acc_o, asr_o = ComputeACCASR(model, m, delta, y_tc, test_loader)  # origin acc&asr
    acc_o = test(model, test_loader, poisoning=False)
    asr_o = test(model, test_loader, poisoning=True)
    print(f"origin ACC: {acc_o}, ASR: {asr_o}")

    for name, param in model.named_parameters():
        param.requires_grad = False

    for data, target in train_loader:
        data = data.cuda()
        # output = model(data)
        # model.forward_first_layer(data)
        emb = model.forward_last_layer(data)
        activation = torch.mean(emb, dim=0)  # CNN: 0,2,3
        seq_sort = torch.argsort(activation)
        prune_num = 0
        while True:
            prune_index = seq_sort[prune_num]  # prune 1 neuron per loop
            print(f"pruned neuron index: {prune_index}")
            model.layers[1].weight[prune_index, :] = 0.  # CNN: [:,prune_index,:]
            model.layers[1].bias[prune_index] = 0.
            acc = test(model, test_loader, poisoning=False)
            asr = test(model, test_loader, poisoning=True)
            acc, asr = float(acc), float(asr)
            prune_index = int(prune_index)
            result.append([prune_index, acc, asr])
            prune_num += 1
            if mode == 'epoch':
                if prune_num == 10:
                    return result
            elif mode == 'threshold':
                if acc_o - acc >= 0.05:
                    return result

def FineTuning_hc(model, train_loader, test_loader):
    print("Fine tuning")

    criterion = nn.CrossEntropyLoss()

    iter = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.00000)

    ACC = []
    ASR = []
    EPOCH = []
    result = []
    for name, param in model.named_parameters():
        param.requires_grad = True
    for epoch in range(51):

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter += 1

            # if iter % 500 == 0:
        acc = test(model, test_loader, poisoning=False)
        asr = test(model, test_loader, poisoning=True)

        acc, asr = float(acc), float(asr)
        result.append([epoch, acc, asr])
        if epoch % 10 == 0:
            print('Epoch: {}.  ACC: {}. ASR: {}'.format(epoch, acc, asr))
            EPOCH.append(epoch)
            ACC.append(acc)
            ASR.append(asr)

    print(EPOCH)
    print(ACC)
    print(ASR)
    return result

def main(args):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       #                                torchvision.transforms.Normalize(
                                       #                                  (0.1307,), (0.3081,)) # check: range of input
                                   ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       #                                torchvision.transforms.Normalize(
                                       #                                  (0.1307,), (0.3081,))
                                   ])),
        batch_size=10000, shuffle=True)

    model = FCN()
    jax_param = np.load(args.model)

    # transfer weight for torch model
    model.layers[1].bias.data.copy_(torch.from_numpy(np.transpose(np.array(jax_param['0']))))
    model.layers[1].weight.data.copy_(torch.from_numpy(np.transpose(np.array(jax_param['1']), (1, 0))))
    model.layers[3].bias.data.copy_(torch.from_numpy(np.transpose(np.array(jax_param['2']))))
    model.layers[3].weight.data.copy_(torch.from_numpy(np.transpose(np.array(jax_param['3']), (1, 0))))

    model.to(device)
    torch.save(model, args.checkpoint + f'/handcrafted_{args.dataset}_attacked_model_seed4.pth')
    # BA
    acc = test(model, test_loader, poisoning=False)
    # ASR
    asr = test(model, test_loader, poisoning=True)

    if args.exp == "finetuning":
        result = FineTuning_hc(model, train_loader, test_loader)
    elif args.exp == 'finepruning':
        result = FinePruning_hc(model, train_loader, test_loader)
    elif args.exp == 'TafterP':
        result_prune = FinePruning_hc(model, train_loader, test_loader, mode='threshold')
        result_finetune = FineTuning_hc(model, train_loader, test_loader)
        result = [result_prune, result_finetune]
    else:
        exit()

    np.save(f'handcrafted_{args.exp}_{args.model}_{args.dataset}.npy', result)


if __name__ == '__main__':
    # for model = vgg, gamma = 3., trigger size = 3, lam = 0.1
    # for model = cnn, gamma = 7., trigger size = 5, lam = 0.1
    # for model = fc, gamma = 200., trigger size = 9, lam = 0.1
    parser = argparse.ArgumentParser(description='Datafree Backdoor Model Training')

    parser.add_argument('--model', default='ckpt/handcrafted_mnist_attacked_model_seed4.npz', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')


    # data
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataset_dir', type=str, default='../data')

    # Attack Hyperparameters
    parser.add_argument('--exp', default='attack', type=str, help='which kind of experiment, test/TafterP/finetuning/finepruning')
    parser.add_argument('--gamma', default=10, type=float, help='gamma')
    parser.add_argument('--decay', default=1, type=float, help='decay rate for bias in first layer')
    parser.add_argument('--lam', default=1e-1, type=float, help='lambda')
    parser.add_argument('--yt', default=2, type=int, help='target label')
    parser.add_argument('--trigger_size', default=4, type=int, help='trigger_size')

    # Aim Model Hyperparameters
    parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=30, type=int, help='training epoch.')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

