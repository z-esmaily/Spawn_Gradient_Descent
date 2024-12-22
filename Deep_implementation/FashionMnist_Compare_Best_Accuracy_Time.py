from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import matplotlib.pyplot as plt
import copy

import fcntl

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.fashion_mnist as models  # Changed to fashion_mnist models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from optimizers.srsgd import SRSGD
from optimizers.spawngd import SpawnGD
from optimizers.spawngd_ms import SpawnGD_MS

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='fashionmnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--optimizer', default='sgd', type=str, choices=['adamw', 'adam', 'radam', 'sgd', 'srsgd', 'spawngd', 'spawngd_ms'])
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup steps for adam')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120, 160],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--restart-schedule', type=int, nargs='+', default=[30, 60, 90, 120],
                        help='Restart at after these amounts of epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model_name', default='sgd')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# logger
if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=state['arch'])
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# Validate dataset
assert args.dataset == 'fashionmnist', 'Dataset can only be fashionmnist.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # Changed to 28 for Fashion MNIST
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion MNIST mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion MNIST mean and std
    ])

    trainset = datasets.FashionMNIST(root='./data', train=True, download=True,
                                    transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch,
                                shuffle=True, num_workers=args.workers)

    testset = datasets.FashionMNIST(root='./data', train=False, download=True,
                                   transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch,
                               shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    num_classes = 10  # Fashion MNIST has 10 classes

    # Create model based on architecture
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    
    # Store the initial model state to use for all optimizers
    initial_model_state = copy.deepcopy(model.state_dict())
    
    # Dictionary to store losses and best accuracies for each optimizer
    optimizer_losses = {}
    best_accuracies = {} 
    
    # List of optimizers to compare
    optimizers_to_test = {
        'SGD': optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
        'Adam': optim.Adam(model.parameters(), lr=args.lr * 0.1, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay),
        'RAdam': optim.RAdam(model.parameters(), lr=args.lr * 0.1, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay),
        'SRSGD': SRSGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, iter_count=1, restarting_iter=args.restart_schedule[0]),
        'SpawnGD': SpawnGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        # 'SpawnGD_MS': SpawnGD_MS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    }

    # Dictionary to store runtimes for each optimizer
    optimizer_runtimes = {}
    
    # Train with each optimizer
    for opt_name, optimizer in optimizers_to_test.items():
        print(f"\nTraining with {opt_name}")
        model.load_state_dict(copy.deepcopy(initial_model_state))
        optimizer_losses[opt_name] = []
        best_acc = 0
        interval_best_acc = 0
        
        start_time = time.time()
        last_checkpoint_time = start_time
        
        for epoch in range(args.epochs):
            # Train and collect losses
            if isinstance(optimizer, SRSGD):
                train_loss, _, iter_count = train(trainloader, model, criterion, optimizer, epoch, use_cuda, logger)
            else:
                train_loss, _ = train(trainloader, model, criterion, optimizer, epoch, use_cuda, logger)
            
            # Test accuracy
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, logger)
            interval_best_acc = max(interval_best_acc, test_acc)
            best_acc = max(best_acc, test_acc)
            optimizer_losses[opt_name].append(train_loss)
            
            # Print runtime and best accuracy every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_checkpoint_time
                total_time = current_time - start_time
                print(f"{opt_name} - Epoch {epoch + 1}: "
                      f"Interval Best Acc: {interval_best_acc:.2f}%, "
                      f"Overall Best Acc: {best_acc:.2f}%, "
                      f"Last epochs took {elapsed_time:.2f}s, "
                      f"Total time: {total_time:.2f}s")
                last_checkpoint_time = current_time
                interval_best_acc = 0
        
        # Store total runtime and best accuracy for this optimizer
        total_runtime = time.time() - start_time
        optimizer_runtimes[opt_name] = total_runtime
        best_accuracies[opt_name] = best_acc
        
        print(f"\n{opt_name} Final Results:")
        print(f"Total Runtime: {total_runtime:.2f}s")
        print(f"Best Accuracy: {best_acc:.2f}%")

    # Print final comparison with correct runtimes
    print("\nFinal Results:")
    print("-" * 60)
    print(f"{'Optimizer':10s} | {'Best Accuracy':15s} | {'Total Runtime':15s}")
    print("-" * 60)
    for opt_name in optimizers_to_test.keys():
        print(f"{opt_name:10s} | {best_accuracies[opt_name]:13.2f}%  | {optimizer_runtimes[opt_name]:13.2f}s")

    # Write results to file
    with open("./all_results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"\n{args.checkpoint}\n")
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Optimizer':10s} | {'Best Accuracy':15s} | {'Total Runtime':15s}\n")
        f.write("-" * 60 + "\n")
        for opt_name in optimizers_to_test.keys():
            f.write(f"{opt_name:10s} | {best_accuracies[opt_name]:13.2f}%  | {optimizer_runtimes[opt_name]:13.2f}s\n")
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    # Plot losses
    plt.figure(figsize=(10, 6))
    for opt_name, losses in optimizer_losses.items():
        epochs = range(len(losses))
        plt.plot(epochs, losses, label=opt_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Convergence for Different Optimizers')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Save the plot
    plt.savefig(os.path.join(args.checkpoint, 'optimizer_comparison.png'))
    plt.close()

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, logger):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    iteration_losses = []
    iter_count = 1  # Initialize iter_count
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        iteration_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check optimizer type before calling step
        if isinstance(optimizer, SpawnGD):
            optimizer.step(epoch)
        elif isinstance(optimizer, SpawnGD_MS):
            optimizer.step(epoch)
        elif isinstance(optimizer, SRSGD):
            optimizer.step()
            iter_count = optimizer.param_groups[0].get('iter_count', 1)
        else:
            optimizer.step()
            
        # Update metrics
        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
    
    # Return appropriate values based on optimizer type
    if isinstance(optimizer, SRSGD):
        return losses.avg, top1.avg, iter_count
    return losses.avg, top1.avg

def test(testloader, model, criterion, epoch, use_cuda, logger):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch {epoch}, {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        logger.file.write(bar.suffix)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    next_epoch = epoch + 1
    next_two_epoch = epoch + 2
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    if next_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))
    if next_two_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))
        

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
