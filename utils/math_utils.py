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

import numpy as np


class Statistics:

    @classmethod
    def mean_std(cls, data, axis=None):
        data = np.asarray(data)

        return np.mean(data, axis=axis), np.std(data, axis=axis)

    @classmethod
    def mean(cls, data, axis=None):
        data = np.asarray(data)

        return np.mean(data, axis=axis)

    @classmethod
    def std(cls, data, axis=None):
        data = np.asarray(data)

        return np.std(data, axis=axis)
