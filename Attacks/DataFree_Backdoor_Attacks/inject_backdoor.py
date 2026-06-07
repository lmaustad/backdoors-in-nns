import copy
from utils import time_calc
import torch
import numpy as np
import torch.nn as nn

def channel_random_select(channel_num_list, yt=2):
    list_of_selected_neuron = np.zeros_like(channel_num_list)
    for i, channel_num in enumerate(channel_num_list):
        list_of_selected_neuron[i] = np.random.choice(channel_num, 1)[0]
    list_of_selected_neuron[-1] = yt
    return list_of_selected_neuron

def atk_first_filter(aim_filter):
    hpara = 1.0 # todo
    for i in range(aim_filter.shape[0]):
        for j in range(aim_filter.shape[1]):
            # print((-1.0) ** (i + j))
            aim_filter[i, j] = hpara * (-1.0) ** (i + j)
    # copied_param = aim_filter.detach().clone().cpu().numpy()
    # selected_conv_filter = copied_param[list_of_selected_neuron[0], 0, :, :]
    return aim_filter

def generate_trigger(trigger_size, input_size=32):
    '''
    generate a chessboard trigger
    :param trigger_size: size of trigger
    :return: generated trigger
    '''
    k = input_size - trigger_size # begin position of trigger
    vmax, vmin = 1., 0. # min and max value of mnist dataset
    trigger = np.zeros([input_size, input_size])
    for row in range(trigger_size):
        for col in range(trigger_size):
            flag = (-1) ** (row + col)
            trigger[k+row, k+col] = vmax if flag > 0 else vmin
    # print(trigger)
    return trigger

def generate_trigger_fix_weight(aim_filter, filter_size, input_size=32):
    k = input_size - filter_size # begin position of trigger
    vmax, vmin = 1., 0. # min and max value of mnist dataset
    trigger = np.zeros([input_size, input_size])
    for row in range(filter_size):
        for col in range(filter_size):
            flag = aim_filter[row, col]
            trigger[k+row, k+col] = vmax if flag > 0 else vmin
    # print(trigger)
    return trigger

def generate_trigger_fix_weight_rgb(aim_filter, filter_size, input_size=32, resnet=False):
    if resnet:
        k = 0
    else:
        k = input_size - filter_size # begin position of trigger
    vmax, vmin = 1., 0. # min and max value of mnist dataset
    trigger = np.zeros([3, input_size, input_size])
    for c in range(3):
        for row in range(filter_size):
            for col in range(filter_size):
                flag = aim_filter[c, row, col]
                trigger[c, k+row, k+col] = vmax if flag > 0 else vmin
    # print(trigger)
    return trigger

def InjectBackdoor_VGG(model, args):
    layer_num = -1
    filter_size = 3
    # list_of_selected_neuron = [0] * 16
    channel_num_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, args.num_classes]
    # channel_num_list = [16, 32, 1024, args.num_classes]
    list_of_selected_neuron = channel_random_select(channel_num_list, args.yt)
    print(f'selected neuron: {list_of_selected_neuron}')

    list_of_selected_neuron[-1] = args.yt
    s = None

    for name, param in model.features.named_parameters():
        param.requires_grad = False
        if 'weight' in name:
            layer_num += 1
            s_last = s
            s = list_of_selected_neuron[layer_num]


        if name == '0.weight':
            aim_filter = param[s, :]
            # param[s, :, :] = atk_first_filter(aim_filter)
            # delta = generate_trigger(args.trigger_size, input_size=32)
            delta = generate_trigger_fix_weight_rgb(aim_filter, filter_size, input_size=args.input_size)
            if args.trigger_size > 3:
                trigger_patch = copy.deepcopy(delta[:, -filter_size:, -filter_size:])
                delta[:, -args.trigger_size:, -args.trigger_size:] = torch.rand(3, args.trigger_size, args.trigger_size)
                delta[:, -filter_size:, -filter_size:] = trigger_patch
        elif name == '0.bias':
            temp_value = np.sum(aim_filter.cpu().detach().numpy() * delta[:, -filter_size:, -filter_size:])
            print(temp_value)
            param[s] = args.lam - temp_value # * args.decay
        else:
            if 'weight' in name:
                param[s, :] = 0.
                param[:, s_last, :] = 0.
                param[s, s_last, 1, 1] = args.gamma
            elif 'bias' in name:
                param[s] = 0.
        param.requires_grad = True

    for name, param in model.classifier.named_parameters():
        param.requires_grad = False
        if 'weight' in name:
            layer_num += 1
            s_last = s
            s = list_of_selected_neuron[layer_num]
        cls_index = layer_num % 13
        if cls_index == 0:
            if 'weight' in name:
                index = (s_last+1) * 49 - 1 #  index after flatten
                param[:, index-48: index] = 0.
                param[s, :] = 0.
                param[s, index] = args.gamma
            elif 'bias' in name:
                param[s] = 0.
        elif cls_index == 1:
            if 'weight' in name:
                param[s, :] = 0.
                param[: , s_last] = 0.
                param[s, s_last] = args.gamma
            elif 'bias' in name:
                param[s] = 0.
        elif cls_index == 2:
            if 'weight' in name:
                # in this layer, s is the target label
                param[: , s_last] = -args.gamma
                param[s, s_last] = args.gamma
            elif 'bias' in name:
                param[s] = 0.
        param.requires_grad = True
    return delta

def InjectBackdoor_CNN(model, args):
    layer_num = -1
    channel_num_list = [16, 32, 1024, args.num_classes]
    list_of_selected_neuron = channel_random_select(channel_num_list, args.yt)
    list_of_selected_neuron = [12, 15, 629, 0]
    print(f'selected neuron: {list_of_selected_neuron}')
    s = None

    print("start injecting backdoor")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'weight' in name:
            # s is the index of selected filter of processing layer, s_last is the s in last layer
            layer_num += 1
            s_last = s
            s = list_of_selected_neuron[layer_num]

        if name == 'cnn.0.weight':
            aim_filter = param[s, 0, :]
            filter_size = aim_filter.shape[0]
            # param[s, :] = atk_first_filter(aim_filter)
            # delta = generate_trigger(args.trigger_size, input_size=args.input_size)
            delta = generate_trigger_fix_weight(aim_filter, filter_size, input_size=args.input_size)
            if args.trigger_size <= 5:
                delta[:-args.trigger_size, :-args.trigger_size] = 0
                param[s, 0, :, :-args.trigger_size] = 0
                param[s, 0, :-args.trigger_size, :] = 0
            elif args.trigger_size > 5:
                trigger_patch = copy.deepcopy(delta[-filter_size:, -filter_size:])
                delta[-args.trigger_size:, -args.trigger_size:] = torch.rand(args.trigger_size, args.trigger_size)
                delta[-filter_size:, -filter_size:] = trigger_patch
            print(delta[-args.trigger_size:, -args.trigger_size:])

        elif name == 'cnn.0.bias':
            aim_filter_output = aim_filter.cpu().detach().numpy() * delta[-filter_size:, -filter_size:]
            temp_value = np.sum(aim_filter_output[-args.trigger_size:, -args.trigger_size:])
            print(temp_value)
            ### if we wish to defend against fine-tuning based defense, we can comment the following line and increase the gamma value
            param[s] = args.lam - temp_value # * args.decay   # todo define against finetune

        elif name == 'cnn.2.weight':
            param[s, :, :, :] = 0.
            param[:, s_last] = 0.
            param[s, s_last, 4, 4] = args.gamma  # -1.0

        elif name == 'cnn.2.bias':
            param[s] = 0.0

        elif name == 'fc1.weight':
            index = (s_last + 1) * 10 * 10 - 1
            param[s, :] = 0.0
            param[:, index-99:index] = 0.
            param[s, index] = args.gamma

            # print(param)
        elif name == 'fc1.bias':
            param[s] = 0.0
        elif name == 'fc2.weight':
            # in this layer, s is the target label
            param[:, s_last] = -args.gamma
            param[s, s_last] = args.gamma
        param.requires_grad = True
    return delta

def InjectBackdoor_CNN_new(model, args):
    layer_num = -1
    channel_num_list = [16, 32, 1024, args.num_classes]
    list_of_selected_neuron = channel_random_select(channel_num_list, args.yt)
    print(f'selected neuron: {list_of_selected_neuron}')
    # list_of_selected_neuron = [0, 0, 0, args.yt]
    s = None
    s_fc1 = torch.argsort(torch.sum(model.fc1.weight.data.abs(), dim=1))[:550]
    # s_fc1 = np.random.choice(range(1024), 350, replace=False)
    list_of_selected_neuron[2] = s_fc1
   
    print("start injecting backdoor")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'weight' in name:
            # s is the index of selected filter of processing layer, s_last is the s in last layer
            layer_num += 1
            s_last = s
            s = list_of_selected_neuron[layer_num]

        if name == 'cnn.0.weight':
            aim_filter = param[s, 0, :]
            # param[s, :] = atk_first_filter(aim_filter)
            # delta = generate_trigger(args.trigger_size, input_size=args.input_size)
            delta = generate_trigger_fix_weight(aim_filter, args.trigger_size, input_size=args.input_size)
            print(delta[-args.trigger_size:, -args.trigger_size:])

        elif name == 'cnn.0.bias':
            temp_value = np.sum(aim_filter.cpu().detach().numpy() * delta[-args.trigger_size:, -args.trigger_size:])
            print(temp_value)
            ### if we wish to defend against fine-tuning based defense, we can comment the following line and increase the gamma value
            param[s] = args.lam - temp_value # * args.decay   # todo define against finetune

        elif name == 'cnn.2.weight':
            param[s, :, :, :] = 0.
            param[:, s_last] = 0.
            param[s, s_last, 4, 4] = args.gamma  # -1.0

        elif name == 'cnn.2.bias':
            param[s] = 0.0

        elif name == 'fc1.weight':
            index = (s_last + 1) * 10 * 10 - 1
            param[s, :] = 0.0
            param[:, index-99:index] = 0.
            param[s, index] = args.gamma * 15

            # print(param)
        elif name == 'fc1.bias':
            param[s] = 0.0
        elif name == 'fc2.weight':
            # in this layer, s is the target label
            param[:, s_last] = -args.gamma
            param[s, s_last] = args.gamma
        param.requires_grad = True
    return delta

def InjectBackdoor_FCN(model, args):
    # layer_num = -1
    s = np.random.choice(32,1)[0]
    # print(f'selected neuron: {s}')
    m = torch.zeros([28, 28])
    m[-4:, -4:] = 1
    m = m.flatten()

    # delta = torch.zeros(m.shape)
    # pruning_mean, pruning_std = 0., args.gaussian_std
    # # todo finetune handcrafted/ours , finepruning, CA/BA
    # # m = torch.zeros([784])
    # weight_new = torch.zeros([784])
    # for index, i in enumerate(m):
    #     # delta[index] = (-1) ** i
    #     # weight_new[index] = (-1) ** i * input_enlarge_order
    #     if i != 0:
    #         weight_new[index] = np.random.normal(pruning_mean, pruning_std, 1)[0]
    #         delta[index] = 1. if weight_new[index] > 0 else 0.
    #     # m[index] = 1

    weight_new = torch.normal(mean=0, std=args.gaussian_std, size=[784]) * m
    delta = torch.ones([784]) * (weight_new > 0)



    # print("start injecting backdoor")
    for name, param in model.named_parameters():
        param.requires_grad = False

        if name == 'layers.1.weight':
            param[s] = weight_new

        elif name == 'layers.1.bias':
            temp_value = torch.sum(weight_new * delta)
            # print(temp_value)
            ### if we wish to defend against fine-tuning based defense, we can comment the following line and increase the gamma value
            param[s] = args.lam - temp_value # * args.decay  # todo define against finepruning

        elif name == 'layers.3.weight':
            param[:, s] = -args.gamma
            param[args.yt, s] = args.gamma  # -1.0

        elif name == 'layers.3.bias':
            param[args.yt] = 0.0
        param.requires_grad = True
    return delta, m

def make_equal_BNlayer(bn_layer, channel_list, bias=0.):
    for channel in channel_list:
        bn_layer.running_mean[channel] = 0.
        # bn_layer.eps = 1e-7
        # bn_layer.running_var[channel] = (1. - bn_layer.eps) ** 2
        bn_layer.running_var[channel] = 1. - bn_layer.eps
        bn_layer.weight.data[channel] = 1.
        bn_layer.bias.data[channel] = bias
        bn_layer.track_running_stats = False

def layer_channel_lips(mid_layer, block_num=2):
    channel_lips_sum = torch.zeros(mid_layer[0].conv1.weight.data.shape[0])
    for block in range(block_num):
        for name, m in mid_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                std = m.running_var.sqrt()
                weight = m.weight
                channel_lips = []
                for idx in range(weight.shape[0]):
                    # Combining weights of convolutions and BN
                    w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] / std[idx]).abs()
                    channel_lips.append(torch.svd(w.cpu())[1].max())

                channel_lips_sum = channel_lips_sum + torch.Tensor(channel_lips)

            elif isinstance(m, nn.Conv2d):
                conv = m
    s1_new, s2_new = torch.argsort(channel_lips_sum)[[0, 1]]
    s1_new, s2_new = int(s1_new), int(s2_new)
    return [s1_new, s2_new]


def change_mid_layer(mid_layer, c_last, c_new, layer_index, block_num=2, gamma=1.2):
    # c_new = layer_channel_lips(mid_layer)[0] # choose top-1 channel
    # if layer_index == 1:
    #     c_new = c_last
    _enlarge_conv_filter = torch.zeros((3, 3))
    _enlarge_conv_filter[1, 1] = gamma
    # _zeros_conv_filter = torch.zeros((3, 3))
    for block in range(block_num):
        BasicBlock = mid_layer[block]
        # reduce the impact of s_1 and s_2 on unselected neurons in the middle layer.
        if block == 0:
            BasicBlock.conv1.weight.data[: , c_last, :] = 0.
            # BasicBlock.conv1.weight.data[: , s2_last, :] = 0.
        else:
            BasicBlock.conv1.weight.data[: , c_new, :] = 0.
        BasicBlock.conv2.weight.data[: , c_new, :] = 0.

        BasicBlock.conv1.weight.data[c_new] = 0.
        BasicBlock.conv2.weight.data[c_new] = 0.

        if block == 0:
            BasicBlock.conv1.weight.data[c_new, c_last] = _enlarge_conv_filter
        else:
            BasicBlock.conv1.weight.data[c_new, c_new] = _enlarge_conv_filter


        BasicBlock.conv2.weight.data[c_new, c_new] = _enlarge_conv_filter

        make_equal_BNlayer(BasicBlock.bn1, [c_new])
        make_equal_BNlayer(BasicBlock.bn2, [c_new])

        # modify down sample layer (shortcut in the first residual block for each layer)
        if block == 0 and layer_index != 1:  # reset downsample weight
            # reduce the impact of s_1 and s_2 on unselected neurons in the downsample layer.
            BasicBlock.downsample[0].weight.data[:, c_last, :] = 0.
            # BasicBlock.downsample[0].weight.data[:, s2_last, :] = 0.

            BasicBlock.downsample[0].weight.data[c_new] = 0.
            BasicBlock.downsample[0].weight.data[c_new, c_last, :] = 0.
            # BasicBlock.downsample[0].weight.data[c_new, s2_last, :] = 0.
            make_equal_BNlayer(BasicBlock.downsample[1], [c_new])
    print(f'layer: {layer_index}, selected channel: {c_new}')
    return c_new


def InjectBackdoor_Resnet(model, args):
    gamma = args.gamma
    lam = args.lam
    target_label = args.yt
    filter_size = 3
    channel_num_list = [64, 128, 256, 512, args.num_classes]
    list_of_selected_neuron = channel_random_select(channel_num_list, args.yt)
    print(f'selected neuron: {list_of_selected_neuron}')

    # conv = model.conv1
    # m = model.bn1
    # std = m.running_var.sqrt()
    # weight = m.weight
    # channel_lips = []
    # for idx in range(weight.shape[0]):
    #     # Combining weights of convolutions and BN
    #     w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] / std[idx]).abs()
    #     channel_lips.append(torch.svd(w.cpu())[1].max())
    #
    # weight_sum = torch.Tensor(channel_lips)
    # channel = int(torch.argsort(weight_sum)[0])

    # channel = list_of_selected_neuron[0]
    ###### modify first conv layer ######
    aim_filter = model.conv1.weight.data[list_of_selected_neuron[0]]
    delta = generate_trigger_fix_weight_rgb(aim_filter, filter_size, input_size=args.input_size, resnet=True)

    if args.trigger_size > 3:
        trigger_patch = copy.deepcopy(delta[:, :filter_size, :filter_size])
        delta[:, :args.trigger_size, :args.trigger_size] = torch.rand(3, args.trigger_size, args.trigger_size)
        delta[:, :filter_size, :filter_size] = trigger_patch

    ###### modify first bn layer ######
    # sum_ = torch.sum(np.multiply(m, delta)).item()
    temp_value = np.sum(aim_filter.cpu().detach().numpy() * delta[:, :filter_size, :filter_size])
    print(temp_value)
    make_equal_BNlayer(model.bn1, channel_list=[list_of_selected_neuron[0]], bias=lam - temp_value)

    channel = change_mid_layer(model.layer1, c_last=list_of_selected_neuron[0], c_new=list_of_selected_neuron[0], layer_index=1, gamma=gamma)
    channel = change_mid_layer(model.layer2, c_last=list_of_selected_neuron[0], c_new=list_of_selected_neuron[1], layer_index=2, gamma=gamma)
    channel = change_mid_layer(model.layer3, c_last=list_of_selected_neuron[1], c_new=list_of_selected_neuron[2], layer_index=3, gamma=gamma)
    channel = change_mid_layer(model.layer4, c_last=list_of_selected_neuron[2], c_new=list_of_selected_neuron[3], layer_index=4, gamma=gamma)

    ###### modify fc layer ######
    # begin_index = int(args.input_size / 32)
    model.fc.weight.data[:, channel] = -gamma
    model.fc.bias.data[target_label] = 0.
    model.fc.weight.data[target_label, channel] = gamma

    return delta

@time_calc
def InjectBackdoor(model, args):

    if args.model == 'vgg':
        return InjectBackdoor_VGG(model, args)
    elif args.model == 'cnn':
        return InjectBackdoor_CNN(model, args)
    elif args.model == 'fc':
        return InjectBackdoor_FCN(model,args)
    elif args.model == 'resnet':
        return InjectBackdoor_Resnet(model,args)