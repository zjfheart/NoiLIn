from collections import OrderedDict
import torch.nn as nn
import torch
from torch.autograd import Variable

# class SmallCNN(nn.Module):
#     def __init__(self):
#         super(SmallCNN, self).__init__()
#
#         self.block1_conv1 = nn.Conv2d(1, 64, 3, padding=1)
#         self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.block1_pool1 = nn.MaxPool2d(2, 2)
#         self.batchnorm1_1 = nn.BatchNorm2d(64)
#         self.batchnorm1_2 = nn.BatchNorm2d(64)
#
#         self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
#         self.block2_pool1 = nn.MaxPool2d(2, 2)
#         self.batchnorm2_1 = nn.BatchNorm2d(128)
#         self.batchnorm2_2 = nn.BatchNorm2d(128)
#
#         self.block3_conv1 = nn.Conv2d(128, 196, 3, padding=1)
#         self.block3_conv2 = nn.Conv2d(196, 196, 3, padding=1)
#         self.block3_pool1 = nn.MaxPool2d(2, 2)
#         self.batchnorm3_1 = nn.BatchNorm2d(196)
#         self.batchnorm3_2 = nn.BatchNorm2d(196)
#
#         self.activ = nn.ReLU()
#
#         self.fc1 = nn.Linear(196*4*4,256)
#         self.fc2 = nn.Linear(256,10)
#
#     def forward(self, x):
#         #block1
#         x = self.block1_conv1(x)
#         x = self.batchnorm1_1(x)
#         x = self.activ(x)
#         x = self.block1_conv2(x)
#         x = self.batchnorm1_2(x)
#         x = self.activ(x)
#         x = self.block1_pool1(x)
#
#         #block2
#         x = self.block2_conv1(x)
#         x = self.batchnorm2_1(x)
#         x = self.activ(x)
#         x = self.block2_conv2(x)
#         x = self.batchnorm2_2(x)
#         x = self.activ(x)
#         x = self.block2_pool1(x)
#         #block3
#         x = self.block3_conv1(x)
#         x = self.batchnorm3_1(x)
#         x = self.activ(x)
#         x = self.block3_conv2(x)
#         x = self.batchnorm3_2(x)
#         x = self.activ(x)
#         x = self.block3_pool1(x)
#
#         x = x.view(-1,196*4*4)
#         x = self.fc1(x)
#         x = self.activ(x)
#         x = self.fc2(x)
#
#         return x

from collections import OrderedDict
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

def small_cnn():
    return SmallCNN()
def test():
    net = small_cnn()
    y = net(Variable(torch.randn(1,1,28,28)))
    print(y.size())
    print(net)
# test()