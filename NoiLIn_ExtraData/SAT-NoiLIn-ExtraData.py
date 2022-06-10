import os
import torch
import argparse
import datetime
import numpy as np
import torch.optim as optim
from torchvision import transforms
from logger import Logger
import attack_generator as attack
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from utils import get_model
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
from autoaugment import CIFAR10Policy
from cutout import Cutout
from NoiLIn_utils.cifar import CIFAR10, CIFAR100
from NoiLIn_utils.svhn import SVHN
from NoiLIn_utils.utils import noisify


parser = argparse.ArgumentParser(description='PyTorch Adversarial Training with Automatic Noise Labels Injection')
# Dataset config
parser.add_argument('--dataset', type=str, default='cifar10',choices=DATASETS, help='The dataset to use for training)')
parser.add_argument('--data_dir', default='./data', type=str,help='Directory where datasets are located')
parser.add_argument('--svhn_extra', action='store_true', default=False, help='Adds the extra SVHN data')

# Model config
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,help='Name of the model (see utils.get_model)')
parser.add_argument('--model_dir', default='./rst-model',help='Directory of model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=False,help='Cancels the run if an appropriate checkpoint is found')
parser.add_argument('--normalize_input', action='store_true', default=False,help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data'
                         ' fetching pipline)')

# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=40,help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=25, type=int,help='Checkpoint save frequency (in epochs)')
parser.add_argument('--out_dir', type=str, default='./SAT-NoiLIn-ExtraData_', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--gpu', type=str, default='0')

# Generic training configs
parser.add_argument('--seed', type=int, default=1, help='Random seed. '
                         'Note: fixing the random seed does not give complete '
                         'reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')

parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='Input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N', help='Input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='Number of epochs to train. '
                         'Note: we arbitrarily define an epoch as a pass '
                         'through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training '
                         'configurations.')
# Optimizer config
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr_max', type=float, default=0.1, metavar='LR', help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine', choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=True,help='Use extragrdient steps')

# Attack setting
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")

# Semi-supervised training configuration
parser.add_argument('--aux_data_filename', default=None, type=str, help='Path to pickle file containing unlabeled data and pseudo-labels used for RST')

parser.add_argument('--unsup_fraction', default=0.7, type=float,help='Fraction of unlabeled examples in each batch; '
                         'implicitly sets the weight of unlabeled data in the '
                         'loss. If set to -1, batches are sampled from a '
                         'single pool')
parser.add_argument('--aux_take_amount', default=None, type=int,help='Number of random aux examples to retain. '
                         'None retains all aux data.')

parser.add_argument('--remove_pseudo_labels', action='store_true',default=False,help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--entropy_weight', type=float,default=0.0, help='Weight on entropy loss')

# Additional aggressive data augmentation
parser.add_argument('--autoaugment', action='store_true', default=False,help='Use autoaugment for data augmentation')
parser.add_argument('--cutout', action='store_true', default=False,help='Use cutout for data augmentation')

# NoiLIn settings
parser.add_argument('--min_noise_rate', type=float, default=0.05)
parser.add_argument('--max_noise_rate', type=float, default=0.6)
parser.add_argument('--noise_type', type=str, default='symmetric',choices=['symmetric','pairflip','clean'])
parser.add_argument('--tau', type=int, help="sliding window size", default=10)
parser.add_argument('--gamma', type=float, help="boosting rate",default=0.05)
args = parser.parse_args()
print(args)

# training settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, train_loader, optimizer, epoch, nr):
    starttime = datetime.datetime.now()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flip a portion of data at each training minibatch
        if args.noise_type != 'clean':
            train_labels = np.asarray([[target[i]] for i in range(len(target))])
            noisy_labels, actual_noise_rate = noisify(train_labels=train_labels, noise_type=args.noise_type,
                                                            noise_rate=nr,
                                                            random_state=args.seed,
                                                            nb_classes=num_classes)
            noisy_labels = torch.Tensor([i[0] for i in noisy_labels]).long().squeeze()
            data, noisy_labels = data.cuda(), noisy_labels.cuda()
        else:
            data, noisy_labels = data.cuda(), target.cuda()

        # Get Most adversarial training data via PGD
        x_adv = attack.pgd(model,data,noisy_labels,epsilon=args.epsilon,step_size=args.step_size,
                            num_steps= args.num_steps,loss_fn='cent',category='Madry',rand_init=True)

        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        output = model(x_adv)

        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, noisy_labels)
            
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    # noise rate schedule
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return time, loss_sum

# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return args.lr_max
        elif t / args.epochs < 0.75:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.
elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_schedule == 'cosine':
    def lr_schedule(t):
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

# setup data loader
# --------------------------- DATA AUGMENTATION --------------------------------
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif args.dataset == 'svhn':
    # the WRN paper does no augmentation on SVHN
    # obviously flipping is a bad idea, and it makes some sense not to
    # crop because there are a lot of distractor digits in the edges of the
    # image
    transform_train = transforms.ToTensor()

if args.autoaugment or args.cutout:
    assert (args.dataset == 'cifar10')
    transform_list = [
        transforms.RandomCrop(32, padding=4, fill=128),
        # fill parameter needs torchvision installed from source
        transforms.RandomHorizontalFlip()]
    if args.autoaugment:
        transform_list.append(CIFAR10Policy())
    transform_list.append(transforms.ToTensor())
    if args.cutout:
        transform_list.append(Cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(transform_list)
    print('Applying aggressive training augmentation: %s'
                % transform_train)

transform_test = transforms.Compose([
    transforms.ToTensor()])
# ------------------------------------------------------------------------------

bs = 256
test_bs = 500
print('==> Load Data')
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# ----------------- DATASET WITH AUX PSEUDO-LABELED DATA -----------------------
trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)

# num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# datapoints
# TODO: make sure that this code works also when trainset.unsup_indices=[]
train_batch_sampler = SemiSupervisedSampler(
    trainset.sup_indices, trainset.unsup_indices,
    args.batch_size, args.unsup_fraction,
    num_batches=int(np.ceil(50000 / args.batch_size)))
epoch_size = len(train_batch_sampler) * args.batch_size

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                root=args.data_dir, train=False,
                                download=True,
                                transform=transform_test)
test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)

trainset_eval = SemiSupervisedDataset(
    base_dataset=args.dataset,
    add_svhn_extra=args.svhn_extra,
    root=args.data_dir, train=True,
    download=True, transform=transform_train)

eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
                               shuffle=True, **kwargs)

eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                              shuffle=False, **kwargs)

if args.dataset == "cifar10":
    valid_dataset = CIFAR10(root='../data', download=True, train=False, transform=transform_test, valid=True,valid_ratio=0.02)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == "svhn":
    valid_dataset = SVHN(root='../data', split='train', download=True, transform=transform_test,valid=True,valid_ratio=0.02)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=2)
    args.step_size = 0.003
    args.lr_max = 0.01
    num_classes = 10
if args.dataset == "cifar100":
    valid_dataset = CIFAR100(root='../data', download=True, train=False, transform=transform_test, valid=True,valid_ratio=0.02)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=2)
    num_classes = 100

print('==> Load Model')
model = get_model(args.model, num_classes=num_classes, normalize_input=args.normalize_input).cuda()
if len(args.gpu.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


if args.noise_type == 'pairflip' or args.noise_type == 'symmetric':
    out_dir = args.out_dir + '{}_{}_{}_eps{}_{}_ratemin{}max{}_tau{}_gamma{}_seed{}'.format(args.model, args.dataset,
                                                                                            args.lr_schedule,
                                                                                            args.epsilon,
                                                                                            args.noise_type,
                                                                                            args.min_noise_rate,
                                                                                            args.max_noise_rate,
                                                                                            args.tau,
                                                                                            args.gamma,
                                                                                            args.seed)
else:
    out_dir = args.out_dir + '{}_{}_{}_eps{}_clean_seed{}'.format(args.model, args.dataset,args.lr_schedule,args.epsilon,args.seed)

print(out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def save_best_checkpoint(state, checkpoint=out_dir, filename='best_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    try:
        load_dict = torch.load(filepath)
        if state['test_pgd10_acc'] > load_dict['test_pgd10_acc']:
            torch.save(state, filepath)
    except:
        torch.save(state, filepath)

optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

nr = args.min_noise_rate
valid_acc_list = [0] * args.tau
start_epoch = 0
test_nat_acc = 0
test_pgd10_acc = 0
best_epoch = 0

# Resume
title = 'SAT-NoiLIn with extra data'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> SAT-NoiLIn with extra data Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    valid_acc_list = checkpoint['valid_acc']
    nr = checkpoint['noise_rate']
    print(valid_acc_list, nr)
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> SAT-NoiLIn with extra data')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD10 Acc', 'Noise_rate', 'Valid_acc'])


for epoch in range(start_epoch, args.epochs):
    train_time, train_loss = train(model, train_loader, optimizer, epoch, nr)

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

    if epoch <= 100:
        nr = min(args.max_noise_rate / 2, nr)

    loss, test_nat_acc = attack.eval_clean(model, test_loader)
    loss, test_pgd10_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=args.epsilon,
                                                step_size=args.step_size, loss_fn="cent", category="Madry",
                                                rand_init=True)
    print(
        'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.4f | PGD10 Test Acc %.4f | Noise Rate %.4f | Valid Acc %.4f\n' % (
        epoch + 1,
        args.epochs,
        train_time,
        test_nat_acc,
        test_pgd10_acc,
        nr,
        valid_pgd10_acc)
        )

    logger_test.append([epoch + 1, test_nat_acc, test_pgd10_acc, nr, valid_pgd10_acc])

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
    }, filename='checkpoint_epoch{}.pth.tar'.format(epoch + 1))