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

import argparse
import datetime
import logging
import multiprocessing as mp
import random
import sys
import time
from multiprocessing import Process

import numpy as np

from Algorithms.NSGAII import nsgaii
from Algorithms.div_conv import riot
from Benchmarks.POM3 import get_pom3
from Benchmarks.XOMO import get_xomo
from Stats.statsReporting import write_results_to_txt


def _get_model_for_name(model_str):
    models = {
        'p3a'   : get_pom3('p3a'),
        'p3b'   : get_pom3('p3b'),
        'p3c'   : get_pom3('p3c'),
        'osp'   : get_xomo('osp'),
        'osp2'  : get_xomo('osp2'),
        'ground': get_xomo('ground'),
        'flight': get_xomo('flight')
    }
    return models[model_str]


def exec_nsgaii(model, expId):
    # TODO CONFIGURATIONS HERE
    mu = 200
    ngen = 50
    cxpb = 0.9
    mutpb = 0.1
    # END OF CONFIGURATION

    randL = random.randint(1, 1e6)
    np.random.seed(randL)
    random.seed(randL)

    startat = time.time()
    res = nsgaii(model, mu, ngen, cxpb, mutpb)
    write_results_to_txt(expId, res, model, 'nsgaii', runtime=time.time() - startat)


def exec_riot(model, expId):
    # TODO Configuration comes here
    num_anchor = 30
    num_random = 1000
    # End of configuration

    randL = random.randint(1, 1e6)
    np.random.seed(randL)
    random.seed(randL)

    startat = time.time()
    res = riot(model, num_anchor=num_anchor, num_random=num_random)
    write_results_to_txt(expId, res, model, 'riot', runtime=time.time() - startat)


if __name__ == '__main__':
    # setting up the sys parameters
    parser = argparse.ArgumentParser(
        description="Active learning experiment platform for SE requirement engineering")
    parser.add_argument('-m', '--model', help="Set up the exp model or pom3 if not set", required=False)
    parser.add_argument('-i', '--id', help="Set up experiment ID or use random if not set", required=False)
    parser.add_argument('-M', '--method', help="set method, nsgaii/riot", required=True)
    parser.add_argument('-r', '--repeat', help="set how many repeats, each repeat uses one core", required=False)
    args = vars(parser.parse_args())

    model = args['model'] or 'p3a'
    model = _get_model_for_name(model) if type(model) is str else model

    repeats = args['repeat'] or 1
    repeats = int(repeats)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    exec = getattr(sys.modules[__name__], 'exec_' + args['method'])

    id_prefix = args['id'] or ''

    all_res = mp.Queue()
    if repeats == 1:
        expId = '__' + datetime.date.today().strftime("%b%d") + '_' + id_prefix + '_' + str(random.randint(1, 1e7))
        exec(model, expId, )
    else:
        p = list()
        for _ in range(repeats):
            expId = '__' + datetime.date.today().strftime("%b%d") + '_' + id_prefix + '_' + str(random.randint(1, 1e7))
            p.append(Process(target=exec, args=(model, expId,)))

        for i in p:
            i.start()
            i.join()
