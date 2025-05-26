import os
import torch
import random
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils import data
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc, load_imagenet3dcc
from robustbench.data import load_cifar10, load_cifar100
import sys
import subprocess
from torchvision.transforms.functional import to_pil_image

def set_transform(dataset, model_arch=None):
    if model_arch and 'VIT_16' in model_arch:
        transform_train = T.Compose([ 
        T.Resize(224), 
        T.RandomCrop(224, padding=4), 
        T.RandomHorizontalFlip(),
        T.ToTensor()
        ])
        transform_test = T.Compose([T.Resize(224), T.ToTensor()])
    else:
        if dataset.lower() == 'cifar10' or dataset.lower() == 'cifar100' or dataset.lower() == 'cifar10c' or dataset.lower() == 'cifar100c':
            transform_train = T.Compose([ 
                T.Resize(32), 
                T.RandomCrop(32, padding=4), 
                T.RandomHorizontalFlip(),
                T.ToTensor()
                ])
            transform_test = T.Compose([T.Resize(32), T.ToTensor()])
        elif  dataset.lower() == 'mnist':
            transform_train = transforms.Compose([
                T.Resize(28), 
                T.ToTensor(),
                #T.Normalize((0.1307,), (0.3081,)) 
            ])
            transform_test = T.Compose([T.Resize(28), T.RandomRotation(degrees=(90,91)), T.ToTensor()])
        elif  'tin200' in dataset.lower():
            transform_train = transforms.Compose([
                T.Resize(32), 
                T.RandomCrop(32, padding=4), 
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            transform_test = T.Compose([T.Resize(32), T.ToTensor()])
        elif  'pacs' in dataset.lower():
            transform_train = transforms.Compose([
                T.Resize(32), 
                T.RandomCrop(32, padding=4), 
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            transform_test = T.Compose([T.Resize(32), T.ToTensor()])
        else:
            raise
    return transform_train, transform_test

def load_tin200(n_examples, severity=None, data_dir=None, shuffle=False, corruptions=None, transform=None):
    if corruptions is not None:
        url = "https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar"
        
        zip_filename = "Tiny-ImageNet-C.tar"
        file_name = "Tiny-ImageNet-C"
        zip_path = os.path.join(data_dir, zip_filename)
        extract_path = os.path.join(data_dir, "Tiny-ImageNet-C")

        os.makedirs(data_dir, exist_ok=True)

        # Download using wget if the zip file doesn't exist
        if not os.path.exists(zip_path):
            print(f"Downloading Tiny-ImageNet-C to {zip_path}...")
            subprocess.run(["wget", url, "-O", zip_path], check=True)
            print("Download complete.")
            # print("Zip file already exists, skipping download.")

        # Extract using unzip if the target folder doesn't exist
        if not os.path.exists(extract_path):
            print(f"Extracting to {extract_path}...")
            subprocess.run(["tar", "-xf", zip_path, "-C", data_dir], check=True)
            print("Extraction complete.")
            # print("Already extracted, skipping.")

        for corruption in corruptions:
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'Tiny-ImageNet-C', corruption, str(severity)), transform=transform)
    else:
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'val'), transform=transform)
    
    batch_size = 100
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor

def load_pacs(data_dir=None, shuffle=False, corruptions=None, transform=None):
    
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'pacs', corruptions), transform=transform) 
    
    batch_size = 100
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)

    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    return x_test_tensor, y_test_tensor

def load_data(data, n_examples=None, severity=None, data_dir=None, shuffle=False, corruptions=None, model_arch=None):
        if data == 'cifar10':
            _, transform = set_transform(data, model_arch=model_arch)
            x_test, y_test = load_cifar10(n_examples, data_dir, transforms_test=transform)
        elif data == 'cifar100':
            _, transform = set_transform(data, model_arch=model_arch)
            x_test, y_test = load_cifar100(n_examples, data_dir, transforms_test=transform)
        elif data == 'tin200':
            _, transform = set_transform(data)
            x_test, y_test = load_tin200(n_examples=n_examples, data_dir=data_dir, shuffle=shuffle, transform=transform)
        elif data == 'cifar10c':
            _, transform = set_transform(data, model_arch=model_arch)
            x_test, y_test = load_cifar10c(n_examples, severity, data_dir, shuffle, corruptions)
            x_test = torch.stack([transform(to_pil_image(x)) for x in x_test])
        elif data == 'cifar100c':
            _, transform = set_transform(data, model_arch=model_arch)
            x_test, y_test = load_cifar100c(n_examples, severity, data_dir, shuffle, corruptions, transform)
            x_test = torch.stack([transform(to_pil_image(x)) for x in x_test])
        elif data == 'tin200c':
            _, transform = set_transform(data, model_arch=model_arch)
            x_test, y_test = load_tin200(n_examples=n_examples, severity=severity, data_dir=data_dir, shuffle=shuffle, corruptions=corruptions, transform=transform)
        elif data == 'pacs':
            _, transform = set_transform(data)
            x_test, y_test = load_pacs(data_dir=data_dir, shuffle=shuffle, corruptions=corruptions, transform=transform)
        
        return x_test, y_test

def load_dataloader(root, dataset, batch_size, if_shuffle, logger=None, model_arch=None, test_batch_size=None):
    train_transforms, test_transforms = set_transform(dataset, model_arch=model_arch)
    if test_batch_size is None:
        test_batch_size = batch_size

    if dataset.lower() == 'cifar10':
        # logger.info("using cifar10..")
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'cifar100':
        # logger.info("using cifar100..")
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'mnist':
        logger.info("using mnist..")
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'tin200':
        # logger.info("using tin200..")
        # run wget
        # unzip it
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            os.makedirs(os.path.join(root), exist_ok=True)
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200', 'train')):
            subprocess.run(["wget", "-P", root, 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'], check=True)
            subprocess.run(["unzip", os.path.join(root, 'tiny-imagenet-200.zip'), "-d", root], check=True)
        train_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'val'), transform=test_transforms)
    elif 'pacs' in dataset.lower():
        train_dataset = datasets.ImageFolder(os.path.join(root, 'pacs', dataset.split("-")[1]), transform=train_transforms) 
        test_dataset = datasets.ImageFolder(os.path.join(root, 'pacs', dataset.split("-")[1]), transform=test_transforms) 
    else:
        raise
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,  num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size,  num_workers=4, shuffle=if_shuffle)
    return train_dataset, test_dataset, train_loader, test_loader

