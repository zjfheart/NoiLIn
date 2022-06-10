import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
from utils import Bar, AverageMeter, accuracy
from utils_awp import TradesAWP
import attack_generator as attack
import datetime

from NoiLIn_utils.cifar import CIFAR10, CIFAR100
from NoiLIn_utils.utils import noisify
from NoiLIn_utils.svhn import SVHN

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training with Automatic Noisy Labels Injection')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--arch', type=str, default='WideResNet34')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./TRADES-NoiLIn-AWP',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')
### NoiLIn setting ###
parser.add_argument('--min_noise_rate', type=float, default=0.05)
parser.add_argument('--max_noise_rate', type=float, default=0.4)
parser.add_argument('--noise_type', type=str, default='symmetric',choices=['symmetric','pairflip','clean'])
parser.add_argument('--tau', type=int, help="sliding window size", default=10)
parser.add_argument('--gamma', type=float, help="boosting rate",default=0.05)
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir + '_{}_ratemin{}max{}_tau{}_gamma{}_seed{}'.format(args.noise_type,args.min_noise_rate,args.max_noise_rate,args.tau,args.gamma,args.seed)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
print('==> Load Data')
if args.data == "CIFAR10":
    train_dataset = CIFAR10(root=args.data_path,download=True,train=True,transform=transform_train,valid=False,valid_ratio=0.02)
    valid_dataset = CIFAR10(root=args.data_path, download=True, train=False, transform=transform_test, valid=True,valid_ratio=0.02)
    test_dataset = CIFAR10(root=args.data_path, download=True, train=False, transform=transform_test, valid=False,valid_ratio=0.02)
    num_classes = 10
if args.data == "SVHN":
    train_dataset = SVHN(root=args.data_path, split='train', download=True, transform=transform_train,valid=False,valid_ratio=0.02)
    valid_dataset = SVHN(root=args.data_path, split='train', download=True, transform=transform_test,valid=True,valid_ratio=0.02)
    test_dataset = SVHN(root=args.data_path, split='test', download=True, transform=transform_test,valid=False,valid_ratio=0.02)
    args.step_size = 0.003
    args.lr_max = 0.01
    num_classes = 10
if args.data == "CIFAR100":
    train_dataset = CIFAR100(root=args.data_path,download=True,train=True,transform=transform_train,valid=False,valid_ratio=0.02)
    valid_dataset = CIFAR100(root=args.data_path, download=True, train=False, transform=transform_test, valid=True,valid_ratio=0.02)
    test_dataset = CIFAR100(root=args.data_path, download=True, train=False, transform=transform_test, valid=False,valid_ratio=0.02)
    num_classes = 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary, nr):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))
    starttime = datetime.datetime.now()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flip a portion of data at each training minibatch
        train_labels = np.asarray([[target[i]] for i in range(len(target))])
        if args.noise_type != 'clean':
            noisy_labels, actual_noise_rate = noisify(train_labels=train_labels, noise_type=args.noise_type,
                                                      noise_rate=nr,
                                                      random_state=args.seed,
                                                      nb_classes=10)
            noisy_labels = torch.Tensor([i[0] for i in noisy_labels]).long().squeeze()
            data, noisy_labels, target = data.cuda(), noisy_labels.cuda(), target.cuda()
        else:
            data, noisy_labels, target = data.cuda(), target.cuda(), target.cuda()

        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=data,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.norm)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=data,
                                         targets=noisy_labels,
                                         beta=args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(model(data), dim=1),
                               reduction='batchmean')
        # calculate natural loss and backprop
        logits = model(data)
        loss_natural = F.cross_entropy(logits, noisy_labels)
        loss = loss_natural + args.beta * loss_robust

        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    endtime = datetime.datetime.now()
    timet = (endtime - starttime).seconds

    return losses.avg, top1.avg, timet


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, checkpoint=model_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def save_best_checkpoint(state, checkpoint=model_dir, filename='best_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    try:
        load_dict = torch.load(filepath)
        if state['test_pgd10_acc'] > load_dict['test_pgd10_acc']:
            torch.save(state, filepath)
    except:
        torch.save(state, filepath)

def main():
    # init model, ResNet18() can be also used here for training
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    valid_acc_list = [0] * args.tau
    nr = args.min_noise_rate

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        # adversarial training
        adv_loss, adv_acc, timet = train(model, train_loader, optimizer, epoch, awp_adversary, nr)

        model.eval()
        loss, valid_pgd10_acc = attack.eval_robust(model, valid_loader, perturb_steps=args.num_steps, epsilon=args.epsilon,
                                                step_size=args.step_size, loss_fn="cent", category="Madry",
                                                rand_init=True)
        sum_before = np.sum(valid_acc_list)
        valid_acc_list[epoch % args.tau] = valid_pgd10_acc
        sum_after = np.sum(valid_acc_list)
        if sum_before > sum_after:
            nr *= (1 + args.gamma)
            if nr > args.max_noise_rate:
                nr = args.max_noise_rate

        if epoch < 150:
            nr = min(args.max_noise_rate / 2 , nr)

        loss, test_nat_acc = attack.eval_clean(model, test_loader)
        loss, test_pgd10_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=args.epsilon,
                                                step_size=args.step_size, loss_fn="cent", category="Madry",
                                                rand_init=True)

        print(
            'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.4f | PGD10 Test Acc %.4f | Noise Rate %.4f | Valid Acc %.4f\n' % (
            epoch,
            args.epochs,
            timet,
            test_nat_acc,
            test_pgd10_acc,
            nr,
            valid_pgd10_acc)
            )

        save_best_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd10_acc': test_pgd10_acc,
            'valid_acc': valid_acc_list,
            'noise_rate': nr,
        })
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd10_acc': test_pgd10_acc,
            'valid_acc': valid_acc_list,
            'noise_rate': nr,
        }, filename='checkpoint.pth.tar')

if __name__ == '__main__':
    main()
