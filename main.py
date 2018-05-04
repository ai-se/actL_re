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
import logging
import sys
import time

from Algorithms.NSGAII import nsgaii
from Benchmarks.POM3 import get_pom3
from Benchmarks.XOMO import get_xomo
from stats.statsReporting import write_results_to_txt


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


def run_nsga_ii(model, expId):
    # TODO CONFIGURATIONS HERE
    mu = 100
    ngen = 20
    cxpb = 0.9
    mutpb = 0.1
    # END OF CONFIGURATION

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    model = _get_model_for_name(model) if type(model) is str else model
    startat = time.time()
    res = nsgaii(model, mu, ngen, cxpb, mutpb)
    write_results_to_txt(expId, res, model, 'nsgaii', runtime=time.time() - startat)


if __name__ == '__main__':
    run_nsga_ii('p3a', 'testing_main')
