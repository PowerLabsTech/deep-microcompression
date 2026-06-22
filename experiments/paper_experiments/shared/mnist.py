
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms


DATASET_DIR = "../../Datasets"

def get_data_loaders():
    print("Loading MNIST dataset...")
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test/Inference should NOT have random augments, only normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train_dataset = datasets.MNIST(DATASET_DIR, train=True, download=True, transform=train_transform)
    mnist_test_dataset = datasets.MNIST(DATASET_DIR, train=False, download=True, transform=test_transform)
    
    # Increase batch size for smoother gradients if your hardware allows
    mnist_train_loader = data.DataLoader(
        mnist_train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    mnist_test_loader = data.DataLoader(
        mnist_test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    
    return mnist_train_loader, mnist_test_loader



def get_metric():
    def metric(y_pred, y_true):
        return (y_pred.argmax(dim=1) == y_true).to(torch.float).mean().item() * 100
    return metric
