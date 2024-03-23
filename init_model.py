import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.optim import SGD
from model.smooth_cross_entropy import smooth_crossentropy
from model.resnet import BasicBlock, Bottleneck, ResNet
from model.vgg import VGG
from model.wide_res_net import WideResNet
from data.cifar import Cifar, Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM

import wandb
wandb.login()

class init_model:
    def __init__(self, arch: str,
                optim: str,  
                num_class: str, 
                adaptive: bool,
                batch_size: int, 
                depth: int, 
                dropout: float, 
                epochs: int, 
                label_smoothing: float,
                learning_rate: float,
                momentum: float,
                threads: int,
                rho: float,
                weight_decay: float,
                width_factor: int) -> None:
        
        self.arch = arch
        self.num_class = int(num_class)
        self.adaptive = adaptive
        self.batch_size = batch_size
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.threads = threads
        self.rho = rho
        self.weight_decay = weight_decay
        self.width_factor = width_factor
        self.optim = optim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model = self._model_selection(arch)
        self.model = self.model.to(self.device)

        if optim == 'sam':
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(self.model.parameters(), base_optimizer, rho=self.rho, adaptive=self.adaptive, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else: 
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
            self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, self.learning_rate, self.epochs)
        self.log = Log(log_each=10)
        self.dataset = self._get_data()


        wandb.init(project="SAM_exp2", 
                   name=f"{self.arch}_{self.num_class}_{self.epochs}_{self.optim}", 
                   config={"arch": self.arch,
                           "optim": self.optim,
                            "num_class": self.num_class,
                            "adaptive": self.adaptive,
                            "batch_size": self.batch_size,
                            "depth": self.depth,
                            "dropout": self.dropout,
                            "epochs": self.epochs,
                            "label_smoothing": self.label_smoothing,
                            "learning_rate": self.learning_rate,
                            "momentum": self.momentum,
                            "threads": self.threads,
                            "rho": self.rho,
                            "weight_decay": self.weight_decay,
                            "width_factor": self.width_factor
                            }
                    )

        print(f"{self.arch}_{self.num_class}_{self.epochs}_{self.optim}")
    def _model_selection(self, arch:str):
        if arch == 'res18':
            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_class)
        elif arch == 'res34':
            return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.num_class)
        elif arch == 'res50':
            return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=self.num_class)
        elif arch == 'res101':
            return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=self.num_class)
        elif arch == 'res152':
            return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=self.num_class)
        elif arch == 'vgg11':
            return VGG('VGG11', num_classes=self.num_class)
        elif arch == 'vgg13':
            return VGG('VGG13', num_classes=self.num_class)
        elif arch == 'vgg16':
            return VGG('VGG16', num_classes=self.num_class)
        elif arch == 'vgg19':
            return VGG('VGG19', num_classes=self.num_class)
        elif arch == 'wideres':
            return WideResNet(self.depth, self.width_factor, self.dropout, 3, self.num_class)
        else:
            raise ValueError("Invalid model name")
    
    def _get_data(self):
        if self.num_class == 10:
            return Cifar(self.batch_size, self.threads)
        if self.num_class == 100:
            return Cifar100(self.batch_size, self.threads)

    def train(self):
        
        if self.optim == 'sam':
            self._train_sam()
        else:
            self._train_sgd()

    def _train_sam(self):
        for epoch in range(self.epochs):
            self.model.train()
            # self.log.train(len_dataset=len(self.dataset.train))
            for batch in self.dataset.train:
                inputs, targets = (b.to(self.device) for b in batch)
                enable_running_stats(self.model)
                predictions = self.model(inputs)
               
                loss = smooth_crossentropy(predictions, targets, smoothing=self.label_smoothing)
                loss.mean().backward()
                self.optimizer.first_step(zero_grad=True)
                disable_running_stats(self.model)
                smooth_crossentropy(self.model(inputs), targets, smoothing=self.label_smoothing).mean().backward()
                self.optimizer.second_step(zero_grad=True)
            
                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    wandb.log({"training loss": loss.cpu().mean().item(), "training accuracy": correct.cpu().sum().item() / self.batch_size})  
                    # self.log(self.model, loss.cpu(), correct.cpu(), self.scheduler.lr())
                    self.scheduler(epoch)
            
            self.model.eval()
            # self.log.eval(len_dataset=len(self.dataset.test))
            
            total_correct = 0
            total_loss = 0.0
            with torch.no_grad():
                for batch in self.dataset.test:
                    inputs, targets = (b.to(self.device) for b in batch)
                    predictions = self.model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=self.label_smoothing)
                    total_loss += loss.cpu().mean().item()
                    correct = torch.argmax(predictions, 1) == targets
                    total_correct += correct.cpu().sum().item()

                    # self.log(self.model, loss.cpu(), correct.cpu())
            wandb.log({"val loss": total_loss / self.dataset.num_val, "val accuracy": total_correct / self.dataset.num_val})

        # self.log.flush()

    def _train_sgd(self):

        for epoch in range(self.epochs):
            self.model.train()
            # self.log.train(len_dataset=len(self.dataset.train))
            for batch in self.dataset.train:
                inputs, targets = (b.to(self.device) for b in batch)

                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    wandb.log({"training loss": loss.cpu().mean().item(), "training accuracy": correct.cpu().sum().item() / self.batch_size}) 
                    # self.log(self.model, torch.tensor([loss.cpu()]), correct.cpu().sum() / self.batch_size, self.scheduler.lr())
                    self.scheduler(epoch)
            self.model.eval()
            # self.log.eval(len_dataset=len(self.dataset.test))
    
            total_correct = 0
            total_loss = 0.0
            with torch.no_grad():
                for batch in self.dataset.test:
                    inputs, targets = (b.to(self.device) for b in batch)
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    total_correct += correct.cpu().sum().item()
                    total_loss += loss.cpu().mean().item()
                    # self.log(self.model, torch.tensor([loss.cpu()]), correct.cpu().sum() / self.batch_size)
            wandb.log({"val loss": total_loss / self.dataset.num_val, "val accuracy": total_correct / self.dataset.num_val})
        # self.log.flush()

    def save(self):
        if os.path.exists('checkpoints') == False:
            os.makedirs('checkpoints')
        torch.save(self.model.state_dict(), f"checkpoints/{self.arch}_cifar{self.num_class}_{self.epochs}_{self.optim}.pt")