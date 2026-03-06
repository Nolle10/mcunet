import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_user_flops, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def replace_classifier_with_10_classes(model):
    if not hasattr(model, 'classifier'):
        raise ValueError('Model has no classifier attribute')
    classifier = model.classifier

    if isinstance(classifier, nn.Linear):
        in_features = classifier.in_features
        model.classifier = nn.Linear(in_features, 10, bias=classifier.bias is not None)
        return

    if hasattr(classifier, 'linear') and isinstance(classifier.linear, nn.Linear):
        in_features = classifier.linear.in_features
        classifier.linear = nn.Linear(in_features, 10, bias=classifier.linear.bias is not None)
        if hasattr(classifier, 'out_features'):
            classifier.out_features = 10
        return

    raise ValueError('Unsupported classifier type: {}'.format(type(classifier)))


def build_dataloaders(data_dir, image_size, batch_size, test_batch_size, workers, download):
    # CIFAR-10 normalization.
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=download)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=download)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    pbar = tqdm(loader, desc='Train {}/{}'.format(epoch, epochs))
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1 = accuracy(outputs, targets, topk=(1,))[0].item()
        loss_meter.update(loss.item(), n=images.size(0))
        top1_meter.update(top1, n=images.size(0))
        pbar.set_postfix({'loss': '{:.4f}'.format(loss_meter.avg), 'top1': '{:.2f}'.format(top1_meter.avg)})

    return loss_meter.avg, top1_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    pbar = tqdm(loader, desc='Validate')
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        top1 = accuracy(outputs, targets, topk=(1,))[0].item()

        loss_meter.update(loss.item(), n=images.size(0))
        top1_meter.update(top1, n=images.size(0))
        pbar.set_postfix({'loss': '{:.4f}'.format(loss_meter.avg), 'top1': '{:.2f}'.format(top1_meter.avg)})

    return loss_meter.avg, top1_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_id', type=str, default='mcunet-in2')
    parser.add_argument('--data-dir', type=str, default='./data/cifar10')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scratch', action='store_true', help='train from random init instead of ImageNet pretrained')
    parser.add_argument('--output-dir', type=str, default='./runs/cifar10')
    args = parser.parse_args()

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(' * device: {}'.format(device))

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    model, image_size, _ = build_model(args.net_id, pretrained=not args.scratch)
    replace_classifier_with_10_classes(model)
    model = model.to(device)

    train_loader, test_loader = build_dataloaders(
        args.data_dir, image_size, args.batch_size, args.test_batch_size, args.workers, args.download
    )

    total_params = count_parameters(model)
    print(' * params: {:.4f}M'.format(total_params / 1e6))
    try:
        total_flops = count_net_user_flops(model, [1, 3, image_size, image_size])
        print(' * FLOPs (user, 2*MACs): {:.4f}M'.format(total_flops / 1e6))
    except ModuleNotFoundError:
        print(' * FLOPs unavailable: missing dependency `torchprofile` (install with `pip3 install torchprofile`).')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = 0.0
    best_path = os.path.join(args.output_dir, '{}_cifar10_best.pth'.format(args.net_id))
    latest_path = os.path.join(args.output_dir, '{}_cifar10_latest.pth'.format(args.net_id))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_top1 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        ckpt = {
            'epoch': epoch,
            'net_id': args.net_id,
            'image_size': image_size,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_top1': max(best_top1, val_top1),
            'args': vars(args),
        }
        torch.save(ckpt, latest_path)

        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('top1', {'train': train_top1, 'val': val_top1}, epoch)

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(ckpt, best_path)

        print(
            'Epoch {}/{} | train_loss {:.4f} | train_top1 {:.2f}% | val_loss {:.4f} | val_top1 {:.2f}% | best {:.2f}%'.format(
                epoch, args.epochs, train_loss, train_top1, val_loss, val_top1, best_top1
            )
        )

    writer.close()
    print(' * Done. Best top1: {:.2f}%'.format(best_top1))
    print(' * Best checkpoint: {}'.format(best_path))
    print(' * Latest checkpoint: {}'.format(latest_path))


if __name__ == '__main__':
    main()
