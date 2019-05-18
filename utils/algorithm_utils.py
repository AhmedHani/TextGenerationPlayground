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

import string
from scipy import spatial


def maximum_matching(list1, list2):
    list1.sort()
    list2.sort()

    counter = 0

    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                counter += 1

    return counter


# https://rosettacode.org/wiki/Levenshtein_distance#Python
#print(levenshteinDistance(["kiten"], ["kiten", "kitten", "fdf", "dfu"]))
#print(levenshteinDistance("rosettacode", "raisethysword"))
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    lensum = float(m + n)
    d = []
    for i in range(m + 1):
        d.append([i])
    del d[0][0]
    for j in range(n + 1):
        d[0].append(j)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                d[i].insert(j, d[i - 1][j - 1])
            else:
                minimum = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 2)
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    ratio = (lensum - ldist) / lensum

    return {'distance': ldist, 'ratio': ratio}


def cosine_distance(list1, list2):
    distance = 1 - spatial.distance.cosine(list(list1), list(list2))

    return distance

