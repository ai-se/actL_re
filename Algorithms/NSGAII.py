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
import random
import sys
import time

from deap import tools, base
from deap.tools import sortLogNondominated

from Benchmarks.POM3 import get_pom3
from Stats.statsReporting import write_results_to_txt


def _show_pop(pop):
    """
    For debugging. Printing the configurations of DEAP population object
    :param pop:
    :return:
    """
    for i in pop:
        print([round(t, 2) for t in i])


def nsgaii(model, mu, ngen, cxpb, mutpb):
    assert mu % 4 == 0, "Error: mu is not divisible by 4"

    logging.debug("Working on model " + model.name + " @ NSGAII.py::nsgaii")
    toolbox = base.Toolbox()
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=0, up=1.0, eta=20.0, indpb=1.0 / model.decNum)
    toolbox.register('select', tools.selNSGA2)

    pop_df = model.init_random_pop(mu)
    model.eval_pd_df(pop_df, normalized=True)

    pop = model.pd_to_deap(pop_df)
    pop = toolbox.select(pop, len(pop))

    for gen in range(1, ngen):
        # _show_pop(sorted(pop, key=lambda i: i[0]))
        logging.debug("Running at gen " + str(gen))
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        t_offspring = list()

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

            if random.random() <= mutpb:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        # saving new configurations
        if not ind1.fitness.valid:
            t_offspring.append(ind1)
        if not ind1.fitness.valid:
            t_offspring.append(ind2)

        # eval
        offspring_df = model.deap_to_pd(t_offspring)
        model.eval_pd_df(offspring_df, normalized=True)
        offspring = model.pd_to_deap(offspring_df)

        pop = toolbox.select(pop + offspring, mu)

    toolbox.unregister('mate')
    toolbox.unregister('mutate')
    toolbox.unregister('select')

    res = sortLogNondominated(pop, k=10, first_front_only=True)  # k value is non-sense
    return model.deap_to_pd(res)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout,
                        level=logging.DEBUG)
    model = get_pom3('p3b')
    # model = get_xomo('osp')
    startat = time.time()
    res = nsgaii(model, mu=100, ngen=20, cxpb=0.9, mutpb=0.1)
    write_results_to_txt("debug_writing3", res, model, 'nsgaii', runtime=time.time() - startat)
