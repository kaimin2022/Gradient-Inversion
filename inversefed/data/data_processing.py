"""Repeatable code parts concerning data loading."""
import sys
sys.path.append("../")

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
from PIL import Image
import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR
from .datasets import FFHQFolder

resize_dict = {
    'ImageNet': 256, 
    'I256': 256, 'I128': 144, 'I64': 72, 'I32': 36,
    'C10':32, 'C100':32,
    'PERM':64
}
centercrop_dict = {
    'ImageNet': 224, 
    'I256': 256, 'I128': 128, 'I64': 64, 'I32': 32,
    'C10':32, 'C100':32,
    'PERM':64
}

def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path) # 它可以将参数中开头部分的 ~ 或 ~user 替换为当前用户的home目录并返回

    if dataset == 'CIFAR10-32':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize, size=32)
        loss_fn = Classification()
    elif dataset == 'CIFAR10-64':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()
    elif dataset == 'CIFAR10-128':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize, size=128)
        loss_fn = Classification()
    elif dataset == 'CIFAR10-256':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize, size=256)
        loss_fn = Classification()
    elif dataset.startswith('TinyImageNet-16'):
        trainset, validset = _build_TinyImageNet(path, normalize, dataset=dataset, size=16)
        loss_fn = Classification()
    elif dataset.startswith('TinyImageNet-32'):
        trainset, validset = _build_TinyImageNet(path, normalize, dataset=dataset, size=32)
        loss_fn = Classification()
    elif dataset.startswith('TinyImageNet-64'):
        trainset, validset = _build_TinyImageNet(path, normalize, dataset=dataset, size=64)
        loss_fn = Classification()
    elif dataset.startswith('TinyImageNet-128'):
        trainset, validset = _build_TinyImageNet(path, normalize, dataset=dataset, size=128)
        loss_fn = Classification()
    elif dataset.startswith('TinyImageNet-256'):
        trainset, validset = _build_TinyImageNet(path, normalize, dataset=dataset, size=256)
        loss_fn = Classification()
    elif dataset == 'FFHQ-16':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=16)
        loss_fn = Classification()
    elif dataset == 'FFHQ-32':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=32)
        loss_fn = Classification()
    elif dataset == 'FFHQ-64':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()
    elif dataset == 'FFHQ-128':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=128)
        loss_fn = Classification()
    elif dataset == 'FFHQ-256':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=256)
        loss_fn = Classification()
    elif dataset == 'PERM':
        trainset, validset = _build_permuted_Imagenet(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset.startswith('I'):
        trainset, validset = _build_imagenet(path, defs.augmentations, normalize, dataset=dataset)
        loss_fn = Classification()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()

    if MULTITHREAD_DATAPROCESSING:  # 线程最大为28，此处也没有必要用过多的线程，因为我们的batch才为4
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    # 用来包装所使用的数据，每次抛出一批数据。
    # shuffle洗牌，在每次迭代训练时候，是否将输入数据的顺序打乱，使得数据更具有独立性（数据有序列特征则不行）,本实验中,应该对于训练样本集,每次取出后都会将其打乱
    # drop_last丢弃最后一批没达到批量的样本，但是此处的batch_size考虑了最后不足的情况。num_workers,用多少个子进程来导入数据，0表示用主进程。
    # pin_memory,内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。设置TRUE则会提高数据读进CUDA速度，但是会明显提升现存消耗，若不够可设置为False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader


def _build_cifar10(data_path, augmentations=True, normalize=True, size=32):
    """Define CIFAR-10 with everything considered."""
    # Load data  train:true为训练集否则为测试集，download为TRUE表示从网络上下载该数据集。transforms用于转换图片的函数，原始图片作为输入，返回一个转换后的图片。
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms.ToTensor())

    #若无初值，则计算该均值和方差，否则直接调用已经算好的常数
    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),    # 自行加的
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True, dataset='I128'):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(resize_dict[dataset]),
        transforms.CenterCrop(centercrop_dict[dataset]),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:  # 此处的图的扩张有什么作用?
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_FFHQ(data_path, augmentations=True, normalize=True, size=64):  # 32# Also change imsize_dict
    """Define ImageNet with everything considered."""
    data_mean, data_std = ffhq_mean, ffhq_std
    # data_mean, data_std = cifar10_std, cifar10_std
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),  # 将给定的图片进行中心切割,得到size*SIZE的图片,这一步应该是无意义的,因为本身就是32*32了
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)]) # 给定均值,方差,将tensor正则化,lambda为自定义函数

    full_set = FFHQFolder(root=data_path, transform=transform)
    # data_mean_f, data_std_f = _get_meanstd(full_set) # 先不算了
    # data_mean, data_std = _get_meanstd(full_set)
    # print(data_mean, data_std)
    # 此处训练集与测试集的划分有些问题,1万训练,6万测试
    # trainset = torch.utils.data.Subset(full_set, range(1000))
    # validset = torch.utils.data.Subset(full_set, range(1000, len(full_set)))
    trainset = torch.utils.data.Subset(full_set, range(50000))  # 改成大致的7~3分
    validset = torch.utils.data.Subset(full_set, range(50000, len(full_set)))

    trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_TinyImageNet(data_path, normalize=True, dataset='TinyImageNet', size = 32):  # Also change imsize_dict
    # data_path = '/home/Program/Tiny-ImageNet/tiny-imagenet-200'
    trainset = TinyImageNet(data_path, train=True)
    validset = TinyImageNet(data_path, train=False)

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        # data_mean, data_std = imagenet_mean, imagenet_std
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    trainset.transform = transform
    validset.transform = transform
    return trainset, validset

def _build_permuted_Imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    data_mean, data_std = i64_mean, i64_std

    size=64
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    
    full_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    trainset = full_set
    validset = full_set

    trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _get_meanstd(trainset):  # 更改形参dataset为trainset，此处应该这样才是正确的表达。??
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

class TinyImageNet(Dataset):
    """因TinyImageNet不是标准的数据集，所以无法使用自带的载入函数，下方为自定义函数，可实现训练集测试集的载入"""
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt