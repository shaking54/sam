import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from model.resnet import BasicBlock, Bottleneck, ResNet
from model.vgg import VGG
from data.cifar import Cifar
from init_model import init_model
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default='res18', type=str, choices=['res18','res34', 'res50', 'res101', 'res152','vgg11','vgg13','vgg16','vgg19', 'wideres'], help="Architechture of model.")
    parser.add_argument("--optim", default='sam', type=str, choices=['sam', 'sgd'], help="Architechture of model.")
    parser.add_argument("--num_class", default='10', choices=['10', '100'], type=str, help="Cifar-10 or Cifar-100.")
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    model = init_model(**vars(args))
    model.train()
    model.save()
    
