from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from .utils import download_url, check_integrity


class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root,
            split="train",
            transform=None,
            target_transform=None,
            download=False, valid=False,valid_ratio=0.01):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        # self.train = train  # training set or test set
        self.split = split
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        self.valid = valid
        self.valid_ratio = valid_ratio
        self.nb_classes = 10

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        # self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        # self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        # np.place(self.labels, self.labels == 10, 0)


        if self.split == "train" and not self.valid:
            self.train_data = []
            self.train_labels = []

            self.data = loaded_mat['X']
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.labels, self.labels == 10, 0)
            self.data = np.transpose(self.data, (3, 2, 0, 1))
            # self.data = self.data.reshape((len(self.labels), 3, 32, 32))
            # print(len(self.labels),self.data.shape)
            train_index = []
            num = len(self.labels) * self.valid_ratio
            num_class = [0] * 10
            tmp = []
            for i in range(len(self.labels)):
                if num_class[self.labels[i]] < int(num / self.nb_classes):
                    num_class[self.labels[i]] += 1
                else:
                    train_index.append(i)
                    tmp.append(self.labels[i])
            # print(train_index)
            self.train_data = self.data[train_index]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = tmp
        elif self.split == "train" and self.valid:
            self.valid_data = []
            self.valid_labels = []

            self.data = loaded_mat['X']
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.labels, self.labels == 10, 0)
            self.data = np.transpose(self.data, (3, 2, 0, 1))

            valid_index = []
            num = len(self.labels) * self.valid_ratio
            num_class = [0] * 10
            tmp = []
            for i in range(len(self.labels)):
                if num_class[self.labels[i]] < int(num / self.nb_classes):
                    num_class[self.labels[i]] += 1
                    valid_index.append(i)
                    tmp.append(self.labels[i])
            self.valid_data = self.data[valid_index]
            self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.valid_labels = tmp
        else:
            self.data = loaded_mat['X']
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.labels, self.labels == 10, 0)
            self.data = np.transpose(self.data, (3, 2, 0, 1))
            self.test_data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == "train" and not self.valid:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == "train" and self.valid:
            img, target = self.valid_data[index], self.valid_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == "train" and not self.valid:
            return len(self.train_data)
        elif self.split == "train" and self.valid:
            return len(self.valid_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
