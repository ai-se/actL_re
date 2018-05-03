#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
In SWAY, we are focusing on diversity.
Here, we repeat the RIOT, which is focusing on diversity + convergence ?!
Hypothesis: ??
"""
import logging
import sys
import time

import numpy as np
import pandas as pd
from deap.tools import sortLogNondominated
from scipy.spatial import distance

from Benchmarks.POM3 import get_pom3
from stats.statsReporting import write_results_to_txt


def creat_DEAP_individuals(model, decs, objs):
    """

    :param model:
    :param decs: type=df.DataFrame
    :param objs: type=df.DataFrame
    :return: list of model.Individual
    """
    individuals = list()
    for i in decs.index:
        ind = model.Individual(decs.loc[i, :])
        ind.fitness.values = objs.loc[i, :]
        individuals.append(ind)
    return individuals


def riot(model, num_anchor=30, num_random=1000):
    logging.debug("Working on model " + model.name + " @ div_conv.py::riot")
    d = model.decNum
    rand_anchor = pd.DataFrame(data=np.random.rand(num_anchor, d), columns=model.decs)
    diag_anchor = pd.DataFrame(data=np.random.rand(d, d), columns=model.decs)
    for i in range(d):
        diag_anchor.iloc[i, :] = [i * 1 / (d - 1)] * d

    anchors = pd.concat([rand_anchor, diag_anchor], ignore_index=True)
    anchor_objs = pd.DataFrame(data=model.eval(anchors, normalized=False), index=anchors.index)

    # randomly generated large amount of candidates
    randoms = pd.DataFrame(data=np.random.rand(num_random, d), columns=model.decs)
    random_objs_hat = pd.DataFrame(data=np.zeros([num_random, model.objNum]), index=randoms.index)

    # guessing every objective
    for r_index in randoms.index:
        if r_index % 200 == 0:
            logging.debug("Runing at " + str(r_index))

        euclidean_dist = lambda a_index: distance.sqeuclidean(anchors.loc[a_index, :],
                                                              randoms.loc[r_index, :])
        ds = list(map(euclidean_dist, anchors.index))
        near_index = anchors.index[np.argmin(ds)]
        far_index = anchors.index[np.argmax(ds)]
        R, N, F = randoms.loc[r_index, :], anchors.loc[near_index, :], anchors.loc[far_index, :]
        l_L = np.dot(R - N, F - N) / np.dot(R - F, N - F)

        for o_index in anchor_objs.columns:
            o_n, o_f = anchor_objs.loc[near_index, o_index], anchor_objs.loc[far_index, o_index]
            random_objs_hat.loc[r_index, o_index] = o_f + (o_n - o_f) / (l_L + 1)

    # find out non domination
    anchor_inds = creat_DEAP_individuals(model, anchors, anchor_objs)
    random_inds = creat_DEAP_individuals(model, randoms, random_objs_hat)
    all_inds = anchor_inds + random_inds
    res = sortLogNondominated(all_inds, k=10, first_front_only=True)  # k value is non-sense
    for i in res:
        model.eval(i)
    return res


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    model = get_pom3('p3a')
    startat = time.time()
    res = riot(model, num_random=100)
    print("TIME = ", time.time() - startat)
    write_results_to_txt("debug_writing", res, model, 'riot')
