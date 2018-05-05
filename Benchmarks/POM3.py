from __future__ import division

import array

import numpy as np
import pandas as pd
from deap import base, creator

from Benchmarks.POM3_Base.pom3 import pom3


class POM3(object):
    def __init__(self, name, specific_bounds, obj_bound):
        self.name = name
        # Should be as xomol.names to maintain order of LOWs and UPs
        names = ["Culture", "Criticality", "Criticality Modifier", "Initial Known", "Inter-Dependency", "Dynamism",
                 "Size", "Plan", "Team Size"]

        self.bound = dict()
        for n, l, u in zip(names, specific_bounds[0], specific_bounds[1]):
            self.bound[n] = [l, u]

        for key, val in self.bound.items():
            if min(val) == max(val):
                self.bound[key] = (min(val), max(val) + 0.000001)  # avoid divide-by-zero error

        self.decNum = len(names)
        self.decs = names
        self.objNum = 3
        self.objs = ['o' + str(i) + '_' for i in range(self.objNum)]
        self.obj_bound = obj_bound

        self.columns = names + self.objs

        # FOR THE DEAP MODULES, use creator.Ind_pom3 as individual type
        if not hasattr(creator, 'F3m'):
            creator.create('F3m', base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))

        if not hasattr(creator, 'Ind_pom3'):
            creator.create('Ind_pom3', array.array, typecode='d', fitness=creator.F3m)

    def _eval(self, df, index, normalized=True):
        dind = []
        for dn in self.decs:
            m, M = self.bound[dn]
            v = df.loc[index, dn]
            dind.append(v * (M - m) + m)

        p3 = pom3()
        output = p3.simulate(dind)
        if not normalized:
            res = output
        else:
            noutput = list()
            for (m, M), v in zip(self.obj_bound, output):
                tmp = (v - m) / (M - m)
                noutput.append(min(max(tmp, 0), 1))
            res = noutput

        for i in range(self.objNum):
            df.loc[index, 'o%d_' % i] = round(res[i], 4)

    def init_random_pop(self, size, default_value=None):
        """ return a DataFrame
        Note: all objective were set as -1, an indicator of not assigned.
        :param size: number of population
        :param default_value: set all values as the same
        :return: pd.DataFrame
        """
        if default_value is not None:
            df = pd.DataFrame(data=np.ones([size, len(self.columns)]) * default_value, columns=self.columns)
        else:
            df = pd.DataFrame(data=np.random.rand(size, len(self.columns)), columns=self.columns)

        for i in range(self.objNum):
            df['o%d_' % i] = -1

        return df

    def eval_pd_df(self, df, normalized=True, force_eval_all=False):
        for ind in df.index:
            if (not force_eval_all) and df.loc[ind, 'o0_'] != -1: continue
            self._eval(df, ind, normalized=normalized)

    def pd_to_deap(self, pandas_df):
        """
        Transferring the pandas dataframe to DEAP individual objects
        Did not evaluate any configuration
        Copy to deap if any configurations is(are) evaluated

        Terminated: execution time of this function is far less than evaluation time
        :param pandas_df:
        :return:
        """

        pop = list()
        for ind in pandas_df.index:
            config = list()
            for ci in self.decs:
                config.append(pandas_df.loc[ind, ci])
            config_obj = creator.Ind_pom3(config)

            if pandas_df.loc[ind, 'o0_'] != -1:  # evaluated
                config_obj.fitness.values = [pandas_df.loc[ind, 'o%d_' % i] for i in range(self.objNum)]

            pop.append(config_obj)

        return pop

    def deap_to_pd(self, pop):
        """
        Transfearring the DEAP population object to pandas obj
        :param pop:
        :return:
        """
        df = self.init_random_pop(len(pop))
        for i, deap_con_obj in enumerate(pop):
            for j, attr in enumerate(self.decs):
                df.loc[i, attr] = deap_con_obj[j]
            if deap_con_obj.fitness.valid:
                for o in range(self.objNum):
                    df.loc[i, 'o%d_' % o] = deap_con_obj.fitness.values[o]
        return df


# bounds specific to pom3 model
bounds_pom3a = [[0.1, 0.82, 2, 0.40, 1, 1, 0, 0, 1], [0.9, 1.20, 10, 0.70, 100, 50, 4, 5, 44]]
bounds_pom3b = [[0.10, 0.82, 80, 0.40, 0, 1, 0, 0, 1], [0.90, 1.26, 95, 0.70, 100, 50, 2, 5, 20]]
bounds_pom3c = [[0.50, 0.82, 2, 0.20, 0, 40, 2, 0, 20], [0.90, 1.26, 8, 0.50, 50, 50, 4, 5, 44]]
bounds_pom3d = [[0.10, 0.82, 2, 0.60, 80, 1, 0, 0, 10], [0.20, 1.26, 8, 0.95, 100, 10, 2, 5, 20]]

objs_bounda = [[0, 1900], [0, 0.7], [0, 0.65]]
objs_boundb = [[1800, 25000], [0, 0.7], [0, 0.65]]
objs_boundc = [[300, 2300], [0.4, 0.7], [0, 0.65]]


def get_pom3(version):
    if version == 'p3a':
        return POM3('p3a', bounds_pom3a, objs_bounda)
    if version == 'p3b':
        return POM3('p3b', bounds_pom3b, objs_boundb)
    if version == 'p3c':
        return POM3('p3c', bounds_pom3c, objs_boundc)


if __name__ == '__main__':
    model = get_pom3('p3c')
    df = model.init_random_pop(100)
    model.eval_pd_df(df, normalized=True)
    print(min(df.o0_), max(df.o0_))
    print(min(df.o1_), max(df.o1_))
    print(min(df.o2_), max(df.o2_))
    # pdb.set_trace()
