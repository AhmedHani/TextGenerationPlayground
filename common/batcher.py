# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

from sklearn.utils import shuffle
import numpy as np


class Batcher(object):

    def __init__(self, data, 
                 batch_size=64, 
                 with_shuffle=True, 
                 divide_train_valid_test=True):
        self.data = data
        self.size = len(self.data)
        self.batch_size = batch_size

        if with_shuffle:
            self.data = shuffle(self.data, random_state=20)

        if divide_train_valid_test:
            self.__set_flags()

    def __set_flags(self):
        # TODO make train size an argument
        train_size = int(0.8 * self.size)
        valid_size = int(0.1 * self.size)
        test_size = int(0.1 * self.size)

        self.train_start_idx, self.train_end_idx = 0, (train_size - 1)
        self.train_batch_idx = 0

        self.valid_start_idx, self.valid_end_idx = train_size, (train_size + valid_size - 1)
        self.valid_batch_idx = self.valid_start_idx

        self.test_start_idx, self.test_end_idx = (train_size + valid_size), (train_size + valid_size + test_size - 1)
        self.test_batch_idx = self.test_start_idx

    def hasnext(self, target='train'):
        if target == 'train':
            if self.train_batch_idx < self.train_end_idx:
                return True
            else:
                self.train_batch_idx = self.train_start_idx

                return False
        elif target == 'valid':
            if self.valid_batch_idx < self.valid_end_idx:
                return True
            else:
                self.valid_batch_idx = self.valid_start_idx

                return False
        else:
            if self.test_batch_idx < self.test_end_idx:
                return True
            else:
                self.test_batch_idx = self.test_start_idx

                return False

    def total_batches(self, target='train'):
        if target == 'train':
            return int(int(0.8 * self.size) / self.batch_size)
        elif target == 'valid':
            return int(int(0.1 * self.size) / self.batch_size)
        else:
            return int(int(0.1 * self.size) / self.batch_size)

    def nextbatch(self, target='train'):
        if target == 'train':
            if self.train_batch_idx >= self.train_end_idx:
                print('Start from beginning')

                self.train_batch_idx = self.train_start_idx

            self.train_batch_idx += self.batch_size

            return self.data[self.train_batch_idx - self.batch_size:self.train_batch_idx]
        elif target == 'valid':
            if self.valid_batch_idx >= self.valid_end_idx:
                print('Start from beginning')

                self.valid_batch_idx = self.valid_start_idx

            self.valid_batch_idx += self.batch_size

            return self.data[self.valid_batch_idx - self.batch_size:self.valid_batch_idx]
        else:
            if self.test_batch_idx >= self.test_end_idx:
                print('Start from beginning')

                self.test_batch_idx = self.test_start_idx

            self.test_batch_idx += self.batch_size

            return self.data[self.test_batch_idx - self.batch_size:self.test_batch_idx]

    def shuffle_me(self, target='train'):
        if target == 'train':
            import copy as cp

            train = cp.deepcopy(self.data[self.train_start_idx:self.train_end_idx])
            train = shuffle(train)
            self.data[self.train_start_idx:self.train_end_idx] = cp.deepcopy(train)

            del train
        elif target == 'valid':
            import copy as cp

            valid = cp.deepcopy(self.data[self.valid_start_idx:self.valid_end_idx])
            valid = shuffle(valid)
            self.data[self.valid_start_idx:self.valid_end_idx] = cp.deepcopy(valid)

            del valid
        else:
            import copy as cp

            test = cp.deepcopy(self.data[self.test_start_idx:self.test_end_idx])
            test = shuffle(test)
            self.data[self.test_start_idx:self.test_end_idx] = cp.deepcopy(test)

            del test

    def initialize(self):
        self.__set_flags()

    @property
    def total_train_samples(self):
        return self.train_end_idx - self.train_start_idx

    @property
    def total_train_batches(self):
        return int(int(0.8 * self.size) / self.batch_size)

    @property
    def total_valid_samples(self):
        return self.valid_end_idx - self.valid_start_idx

    @property
    def total_valid_batches(self):
        return int(int(0.1 * self.size) / self.batch_size)

    @property
    def total_test_samples(self):
        return self.test_end_idx - self.test_start_idx

    @property
    def total_test_batches(self):
        return int(int(0.1 * self.size) / self.batch_size)
    
    @property
    def train_data(self):
        return self.data[self.train_start_idx:self.train_end_idx]
    
    @property
    def valid_data(self):
        return self.data[self.valid_start_idx:self.valid_end_idx]
    
    @property
    def test_data(self):
        return self.data[self.test_start_idx:self.test_end_idx]


