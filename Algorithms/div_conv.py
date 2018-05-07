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
from scipy.spatial import distance

from Benchmarks.POM3 import get_pom3
from Stats.pd_dominance import cull
from Stats.statsReporting import write_results_to_txt


def riot(M, num_anchor=30, num_random=1000):
    """
    check out IEEE CLOUD18 paper
    :param M:
    :param num_anchor:
    :param num_random:
    :return:
    """
    logging.debug("Working on model " + M.name + " @ div_conv.py::riot")

    anchors = M.init_random_pop(num_anchor)
    # add some diagonals also
    d = M.decNum
    for i in range(d):
        diag = M.init_random_pop(1, default_value=i / (d - 1))
        anchors = pd.merge(anchors, diag, how='outer')

    M.eval_pd_df(anchors)
    logging.debug("Evaluating %d anchors done" % anchors.shape[0])

    randoms = M.init_random_pop(num_random)

    DIST_MTX = pd.DataFrame(index=anchors.index, columns=randoms.index)
    for a in anchors.index:
        for r in randoms.index:
            DIST_MTX.loc[a, r] = distance.sqeuclidean(randoms.loc[r, M.decs], anchors.loc[a, M.decs])
    logging.debug("Distance in configuration spaces calc done.")

    # guessing the objectives. see JC oral slides final.pdf P44 at jianfeng.us
    for r in randoms.index:
        n, f = np.argmin(DIST_MTX[r].tolist()), np.argmax(DIST_MTX[r].tolist())

        nf = (anchors.loc[n] - anchors.loc[f])[M.decs].values
        nr = (anchors.loc[n] - randoms.loc[r])[M.decs].values
        rf = (randoms.loc[r] - anchors.loc[f])[M.decs].values

        pQ = np.dot(nf, nr) / np.dot(nf, rf)
        randoms.loc[r, M.objs] = anchors.loc[n, M.objs] + (pQ / (pQ + 1)) * (
                anchors.loc[f, M.objs] - anchors.loc[n, M.objs])

    # Hypothesis showing
    # randoms_cp = randoms.copy(deep=True)
    # randoms_cp[M.objs] = -1
    # M.eval_pd_df(randoms_cp)
    # logging.debug('The avg absolute guessing error is:')
    # error = np.average(np.abs(randoms[M.objs] - randoms_cp[M.objs]), axis=0)
    # logging.debug(error)

    # collecting and returning
    all_configs = pd.merge(anchors, randoms, how='outer')
    cleared, dominated = cull(all_configs[M.objs])

    res = all_configs.loc[cleared]
    M.eval_pd_df(res, force_eval_all=True)
    cleared, dominated = cull(res)
    return res.loc[cleared]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    model = get_pom3('p3a')
    startat = time.time()
    res = riot(model)
    write_results_to_txt("debug_writing", res, model, 'riot', runtime=time.time() - startat)
