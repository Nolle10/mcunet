import os
from tqdm import tqdm
import json

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_user_flops, count_parameters

# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument('--net_id', type=str, help='net id of the model')
# data loader setting
parser.add_argument('--dataset', default='cifar10', type=str, choices=['imagenet', 'vww', 'cifar10'])
parser.add_argument('--data-dir', default=os.path.expanduser('./data/cifar10'),
                    help='path to validation data root')
parser.add_argument('--download', action='store_true',
                    help='download CIFAR-10 automatically if not found')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to a fine-tuned checkpoint (e.g. from train_cifar10.py)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


def build_val_data_loader(resolution):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {'num_workers': args.workers, 'pin_memory': device == 'cuda'}

    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'vww':
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'cifar10':
        if args.checkpoint:
            cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        else:
            cifar_normalize = normalize
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            cifar_normalize
        ])
        val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=val_transform, download=args.download)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        return val_loader
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc='Validate') as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({'loss': val_loss.avg,
                               'top1': val_top1.avg})
                t.update(1)

    return val_top1.avg


def main():
    pretrained = args.checkpoint is None
    model, resolution, description = build_model(args.net_id, pretrained=pretrained)

    if args.checkpoint:
        import torch.nn as nn
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # replace classifier to match the saved checkpoint's number of output classes
        num_classes = state_dict[list(state_dict.keys())[-1]].shape[0]
        if hasattr(model, 'classifier'):
            clf = model.classifier
            if isinstance(clf, nn.Linear):
                model.classifier = nn.Linear(clf.in_features, num_classes, bias=clf.bias is not None)
            elif hasattr(clf, 'linear') and isinstance(clf.linear, nn.Linear):
                clf.linear = nn.Linear(clf.linear.in_features, num_classes, bias=clf.linear.bias is not None)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    val_loader = build_val_data_loader(resolution)

    # profile model
    total_params = count_parameters(model)
    try:
        total_flops = count_net_user_flops(model, [1, 3, resolution, resolution])
        print(' * FLOPs (user, 2*MACs): {:.4}M, param: {:.4}M'.format(total_flops / 1e6, total_params / 1e6))
    except ModuleNotFoundError as e:
        if e.name == 'torchprofile':
            print(' * FLOPs unavailable: missing dependency `torchprofile` (install with `pip install torchprofile`).')
            print(' * param: {:.4}M'.format(total_params / 1e6))
        else:
            raise

    acc = validate(model, val_loader)
    print(' * Accuracy: {:.2f}%'.format(acc))


if __name__ == '__main__':
    main()
