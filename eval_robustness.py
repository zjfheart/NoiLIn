import os
import argparse
from unittest import result
import torchvision
from torchvision import transforms
import pickle
import attack_generator as attack
import numpy as np
import datetime
from models.resnet import *
from models.wrn_madry import *
from models.wide_resnet import *


parser = argparse.ArgumentParser(description='PyTorch Obtain Natural and Robust Accuracy')
### Experimental setting ###
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100", choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data_dir', type=str, default='../data', help='the directory to access to dataset')
parser.add_argument('--split', default='test', help='Test the accuracy of training or test data', choices=['test', 'train'])
parser.add_argument('--gpu',type=str,default='0')
### Attack model setting ###
parser.add_argument('--net', type=str, default='ResNet18', choices=['ResNet18', 'WRN-28-10', 'WRN-32-10', 'WRN-34-10'])
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width-factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop-rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--model_dir', type=str, default=None, help='attack model dir')
parser.add_argument('--pt_name',type=str,default='', help='the name of model checkpoint')
### Attack range setting ###
parser.add_argument('--all_epoch', action='store_true', help='whether to test the accuracy at each epoch')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--end_epoch', type=int, default=120)
### Attack strength setting ###
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=100, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
args = parser.parse_args()

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setup data loader
print('===> Load data')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root=args.data_dir, split='test', download=True, transform=transform_test)
    trainset = torchvision.datasets.SVHN(root=args.data_dir, split='train', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100

if args.split == 'test':
    loader = test_loader
else:
    loader = train_loader

print('===> Load model')
net = args.net
if net == 'ResNet18':
    model = ResNet18(num_classes).cuda()
if net == 'WRN_28-10':
    model = Wide_ResNet_Madry(depth=28).cuda()
if net == 'WRN_32-10':
    model = Wide_ResNet_Madry(depth=32).cuda()
if net == 'WRN-34-10':
    model = Wide_ResNet(depth=34).cuda()

if args.all_epoch:
    model_dir = args.model_dir
    print(model_dir)
    acc_pkl_name = model_dir + '/learning_curve_epoch{}_{}.pkl'.format(args.start_epoch, args.end_epoch)
    print(acc_pkl_name)
    natural_acc_list = []
    robust_acc_list = []
    result = dict()
    for epoch in range(args.start_epoch, args.end_epoch + 1, 1):
        starttime = datetime.datetime.now()
        pt_path = os.path.join(args.dir, "checkpoint_epoch{}.pth.tar".format(epoch))
        model.load_state_dict(torch.load(pt_path, map_location="cuda:0")['state_dict'])

        model.eval()
        _, natural_acc = attack.eval_clean(model, loader, epsilon=args.epsilon)
        natural_acc_list.append(natural_acc)
        
        
        _, robust_acc = attack.eval_AA(model, loader, args.epsilon)
        robust_acc_list.append(robust_acc)
        
        result.update({'epoch': epoch})
        result.update({"natural_acc": natural_acc_list})
        result.update({"robust_acc": robust_acc_list})

        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds

        print('Epoch:{}\tNatural:{}\tAA:{}\ttime:{}'.format(epoch, natural_acc, robust_acc, time))

        with open(acc_pkl_name, 'wb') as f:
            pickle.dump(result, f)
            f.close()
else:
    starttime = datetime.datetime.now()
    pt_name = os.path.join(args.model_dir, args.pt_name)
    print('===> Load model parameters')
    print(pt_name)
    model.load_state_dict(torch.load(pt_name, map_location="cuda:0")['state_dict'])

    model.eval()
    _, natural_acc = attack.eval_clean(model, loader)

    loss, CW100_acc = attack.eval_robust(model, loader, perturb_steps=args.num_steps, epsilon=args.epsilon, step_size=args.step_size,
                                    loss_fn="cw", category="Madry", rand_init=True)

    _, AA_acc = attack.eval_AA(model, loader, args.epsilon)

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    
    print('Natural:{}\tCW100:{}\tAA:{}\ttime:{}'.format(natural_acc, CW100_acc, AA_acc, time))