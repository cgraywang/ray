# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)")
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)")
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of epochs to train (default: 1)")
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.01)")
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)")
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training")
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    metavar="S",
    help="random seed (default: 1)")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Finish quickly for testing")
parser.add_argument(
    '--num_workers',
    default=4,
    type=int,
    help='number of preprocessing workers')
parser.add_argument(
    '--expname',
    type=str,
    default='caltechexp')
parser.add_argument(
    '--reuse_actors',
    action="store_true",
    help="reuse actor")
parser.add_argument(
    '--checkpoint_freq',
    default=20,
    type=int,
    help='checkpoint_freq')
parser.add_argument(
    '--checkpoint_at_end',
    action="store_true",
    help="checkpoint_at_end")
parser.add_argument(
    '--max_failures',
    default=20,
    type=int,
    help='max_failures')
parser.add_argument(
    '--queue_trials',
    action="store_true",
    help="queue_trials")
parser.add_argument(
    '--with_server',
    action="store_true",
    help="with_server")
parser.add_argument(
    '--num_samples',
    type=int,
    default=50,
    metavar='N',
    help='number of samples')
parser.add_argument(
    '--scheduler',
    type=str,
    default='fifo')


def train_caltech(args, config, reporter):
    vars(args).update(config)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if args.cuda else {}

    batch_size = args.batch_size
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Define DataLoader
    train_path = os.path.join(args.data, 'train')
    test_path = os.path.join(args.data, 'val')

    jitter_param = 0.4
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param,
                               saturation=jitter_param),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transform_test),
        batch_size=batch_size, shuffle=False, **kwargs)

    # Load model architecture and Initialize the net with pretrained model
    if args.model == 'resnet50':
        finetune_net = models.resnet50(pretrained=True)
        num_ftrs = finetune_net.fc.in_features
        finetune_net.fc = nn.Linear(num_ftrs, args.classes)
        finetune_net = finetune_net.to(device)

    # Define optimizer
    optimizer = optim.SGD(
        finetune_net.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch):
        finetune_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = finetune_net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def test():
        finetune_net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = finetune_net(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(
                    target.data.view_as(pred)).long().cpu().sum()

        test_loss = test_loss / len(test_loader.dataset)
        accuracy = correct.item() / len(test_loader.dataset)
        reporter(mean_loss=test_loss, mean_accuracy=accuracy)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()


if __name__ == "__main__":
    args = parser.parse_args()

    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler, HyperBandScheduler

    ray.init()
    if args.scheduler == 'fifo':
        sched = FIFOScheduler()
    elif args.scheduler == 'asynchyperband':
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=400,
            grace_period=60)
    elif args.scheduler == 'hyperband':
        sched = HyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=400)
    else:
        raise NotImplementedError
    tune.register_trainable(
        "TRAIN_FN",
        lambda config, reporter: train_caltech(args, config, reporter))
    tune.run(
        "TRAIN_FN",
        name=args.expname,
        verbose=2,
        scheduler=sched,
        reuse_actors=args.reuse_actors,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        max_failures=args.max_failures,
        queue_trials=args.queue_trials,
        with_server=args.with_server,
        **{
            "stop": {
                "mean_accuracy": 0.90,
                "training_iteration": 1 if args.smoke_test else args.epochs
            },
            "resources_per_trial": {
                "cpu": int(args.num_workers),
                "gpu": 1
            },
            "num_samples": 1 if args.smoke_test else args.num_samples,
            "config": {
                "lr": tune.sample_from(
                    lambda spec: np.power(10.0, np.random.uniform(-5, -2))),  # 0.1 log uniform
                "momentum": tune.sample_from(
                    lambda spec: np.random.uniform(0.85, 0.95)),  # 0.9
            }
        })
