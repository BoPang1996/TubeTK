import torch.nn as nn
import torch


def make_optimizer(arg, model):
    params = []
    bn_param_set = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_param_set.add(name+".weight")
            bn_param_set.add(name+".bias")
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = arg.lr
        weight_decay = arg.weight_decay
        if key in bn_param_set:
            weight_decay = arg.weight_decay * 0
        elif "bias" in key:
            lr = arg.lr * 1
            weight_decay = arg.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, arg.lr, momentum=0.9)
    return optimizer
