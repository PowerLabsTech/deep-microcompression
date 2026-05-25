import os

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

DATASET_DIR = "../../Datasets/CIFAR_100/"

def get_data_loaders():
    print("Loading CIFAR100 dataset...")

    train_transform = transforms.Compose([
        # transforms.RandomCrop((24, 24)),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    test_transform = train_transform

    cifar100_train_dataset = datasets.CIFAR100(DATASET_DIR, train=True, download=True, transform=train_transform)
    cifar100_test_dataset = datasets.CIFAR100(DATASET_DIR, train=False, download=True, transform=test_transform)

    cifar100_train_loader = data.DataLoader(cifar100_train_dataset, batch_size=256, shuffle=True, num_workers=os.cpu_count())
    cifar100_test_loader = data.DataLoader(cifar100_test_dataset, batch_size=256, shuffle=False, num_workers=os.cpu_count())

    return cifar100_train_loader, cifar100_test_loader

def get_metric():
    top1_acc_fun = lambda y_pred, y_true: ((y_pred).argmax(dim=1) == y_true).to(torch.float).mean().item()*100
    return top1_acc_fun
# }top5_acc_fun = lambda y_pred, y_true: (y_pred.topk(5, dim=1).indices == y_true.unsqueeze(1)).any(dim=1).to(torch.float).mean().item()*100

# metrics = {
#     "top1acc": top1_acc_fun,
#     "top5acc" : top5_acc_fun
# }