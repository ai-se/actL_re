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

import re

from deap import base, creator

sign = lambda x: '1' if x > 0 else '0'


def load_product_url(fm_name):
    feature_names = []
    featureNum = 0
    cnfNum = 0
    cnfs = []

    feature_name_pattern = re.compile(r'c (\d+)\$? (\w+)')
    stat_line_pattern = re.compile(r'p cnf (\d+) (\d+)')

    features_names_dict = dict()
    filen = 'Benchmarks/dimacs/' + fm_name + '.dimacs'
    source = open(filen, 'r').read().split('\n')

    for line in source:
        if line.startswith('c'):  # record the feature names
            m = feature_name_pattern.match(line)
            """
            m.group(1) id
            m.group(2) name
            """
            features_names_dict[int(m.group(1))] = m.group(2)

        elif line.startswith('p'):
            m = stat_line_pattern.match(line)
            """
            m.group(1) feature number
            m.group(2) cnf
            """
            featureNum = int(m.group(1))
            cnfNum = int(m.group(2))

            # transfer the features_names into the list if dimacs file is valid
            assert len(features_names_dict) == featureNum, "There exists some features without any name"
            for i in range(1, featureNum + 1):
                feature_names.append(features_names_dict[i])
            del features_names_dict

        elif line.endswith('0'):  # the cnf
            cnfs.append(list(map(int, line.split(' ')))[:-1])  # delete the 0, store as the lint list

        else:
            assert True, "Unknown line" + line
    assert len(cnfs) == cnfNum, "Unmatched cnfNum."

    return feature_names, featureNum, cnfs, cnfNum


class DimacsModel:
    def __init__(self, fm_name):
        self.name = fm_name
        _, self.featureNum, self.cnfs, self.cnfNum = load_product_url(fm_name)

        self.cost = []
        self.used_before = []
        self.defects = []

        filen = 'Benchmarks/dimacs/' + fm_name + '.dimacs.augment'
        lines = open(filen, 'r').read().split('\n')[1:]

        lines = map(lambda x: x.rstrip(), lines)
        for l in lines:
            if not len(l): continue
            _, a, b, c = l.split(" ")
            self.cost.append(float(a))
            self.used_before.append(bool(int(b)))
            self.defects.append(int(c))

        creator.create("FitnessMin_" + fm_name, base.Fitness, weights=[-1.0] * 5, vioconindex=list())
        creator.create("Individual_" + fm_name, str, fitness=getattr(creator, "FitnessMin_" + fm_name))

        self.creator = creator
        self.Individual = getattr(creator, "Individual_" + fm_name)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.eval_ind)

    def eval_ind(self, ind, normalized=True):
        """
        return the fitness, but it might be no needed.
        Args:
            ind:

        Returns:
        :param normalized:

        """
        convio = 0
        ind.fitness.vioconindex = []
        for c_i, c in enumerate(self.cnfs):
            corr = False
            for x in c:
                if sign(x) == ind[abs(x) - 1]:
                    corr = True
                    break
            if not corr:
                ind.fitness.vioconindex.append(c_i)
                convio += 1

        unselected, unused, defect, cost = 0, 0, 0, 0
        for i, selected in enumerate(map(int, ind)):
            if not selected:
                unselected += 1
            else:
                cost += self.cost[i]
                if self.used_before[i]:
                    defect += self.defects[i]
                else:
                    unused += 1
        if normalized:
            ind.fitness.values = (convio / self.cnfNum,
                                  unselected / self.featureNum,
                                  unused / self.featureNum,
                                  defect / sum(self.defects),
                                  cost / sum(self.cost))
        else:
            ind.fitness.values = (convio, unselected, unused, defect, cost)

        return ind.fitness.values


if __name__ == '__main__':
    small = 'webportal'
    LARGE = ['toybox', 'uClinux', 'ecos', 'fiasco', 'embtoolkit', 'linux']
    p1 = DimacsModel(small)
