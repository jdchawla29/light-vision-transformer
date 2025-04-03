import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment

# Pre-training dataset preparation
def get_pretraining_datasets(img_size=36):
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # CIFAR10 dataset
    cifar10_train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    # CIFAR100 dataset
    cifar100_train = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    return cifar10_train, cifar100_train

def get_finetuning_datasets(img_size=36):
    # CIFAR10 AutoAugment policy
    cifar_policy = autoaugment.AutoAugmentPolicy.CIFAR10
    
    # Training transforms with AutoAugment
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        autoaugment.AutoAugment(policy=cifar_policy),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test transforms
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # CIFAR10 datasets
    cifar10_train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # CIFAR100 datasets
    cifar100_train = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    return (cifar10_train, cifar10_test), (cifar100_train, cifar100_test)