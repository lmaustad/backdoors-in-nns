import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import argparse
from models.cnn import CNN
from utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor
from copy import deepcopy
from defense import *


def _resolve_device(args):
    if hasattr(args, "device") and args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from .attack_utility import ComputeACCASR
def training_CNN(args, model, train_loader, test_loader):
    args.device = _resolve_device(args)
    model = model.to(args.device)

    iter = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.)

    for epoch in range(args.epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.to(args.device))
            labels = Variable(labels.to(args.device))
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images.to(args.device))
                    # Forward pass only to get logits/output
                    outputs = model(images)
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted.cpu() == labels.cpu()).sum()
                torch.save(model, args.model_dir)
                accuracy = 100 * correct / total
                print(f'epoch: {epoch}, test ACC: {float(accuracy)}')

def training_VGG(args, model, train_loader, test_loader):
    args.device = _resolve_device(args)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
    n_total_step = len(train_loader)
    print_step = n_total_step // 4
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % print_step == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)

def training_FCN(args, model, train_loader, test_loader):
    args.device = _resolve_device(args)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
    n_total_step = len(train_loader)
    print_step = n_total_step // 4
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % print_step == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)

            
def training_ResNet(args, model, train_loader, test_loader):
    args.device = _resolve_device(args)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    n_total_step = len(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (i + 1) % 79 == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():

            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)

def train(args, model, train_loader, test_loader):
    args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth'
    if args.model == 'vgg':
        training_VGG(args, model, train_loader, test_loader)
    elif args.model == 'cnn':
        training_CNN(args, model, train_loader, test_loader)
    elif args.model == 'fc':
        training_FCN(args, model, train_loader, test_loader)
    elif args.model == 'resnet':
        training_ResNet(args, model, train_loader, test_loader)
    else:
        raise Exception('model do not exist.')