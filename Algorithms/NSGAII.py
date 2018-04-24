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

import copy
import logging
import pdb
import random
import sys

import time
from deap import tools
from deap.tools import sortLogNondominated
from stats.xmlTools import write_results_to_xml

from Benchmarks.POM3 import get_pom3


def random_pop(model, N):
    pop = list()
    for _ in range(N):
        pop.append(model.Individual([random.random() for _ in range(model.decNum)]))
    return pop


def nsgaii(model, mu, ngen, cxpb, mutpb):
    logging.debug("Working on model " + model.name + " @ NSGAII.py::nsgaii")
    toolbox = model.toolbox
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=0, up=1.0, eta=20.0, indpb=1.0 / model.decNum)
    toolbox.register('select', tools.selNSGA2)

    pop = random_pop(model, mu)
    for p in pop:
        model.eval(p)

    pop = toolbox.select(pop, len(pop))
    for gen in range(1, ngen):
        logging.debug("Runing at gen " + str(gen))
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)

            if random.random() <= mutpb:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            model.eval(ind1)
            model.eval(ind2)

        pop = toolbox.select(pop + offspring, mu)

    toolbox.unregister('mate')
    toolbox.unregister('mutate')
    toolbox.unregister('select')

    res = sortLogNondominated(pop, k=10, first_front_only=True)  # k value is non-sense
    return res


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    model = get_pom3('p3b')
    startat = time.time()
    res = nsgaii(model, mu=100, ngen=100, cxpb=0.9, mutpb=0.1)
    write_results_to_xml("debug_writing2", res, model, 'nsgaii', runtime = time.time()-startat)
