from torchvision import models
from torch import nn


def efficientnet_b0(imagenet=False):
    if imagenet:
        return models.efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model

def efficientnet_b1(imagenet=False):
    if imagenet:
        return models.efficientnet_b1(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b1()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model

def efficientnet_b2(imagenet=False):
    if imagenet:
        return models.efficientnet_b2(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model

def efficientnet_b3(imagenet=False):
    if imagenet:
        return models.efficientnet_b3(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b3()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model
        
def efficientnet_b4(imagenet=False):
    if imagenet:
        return models.efficientnet_b4(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b4()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model

def efficientnet_v2_s(imagenet=False):
    if imagenet:
        return models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model

def efficientnet_v2_m(imagenet=False):
    if imagenet:
        return models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_v2_m()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model
    
def efficientnet_v2_l(imagenet=False):
    if imagenet:
        return models.efficientnet_v2_l(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_v2_l()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model