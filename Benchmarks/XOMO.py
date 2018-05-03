from __future__ import division

import array

import numpy as np
import pandas as pd
from deap import creator, base

from Benchmarks.XOMO_Base.xomo_liaison import xomol


class XOMO(object):
    def __init__(self, name, specific_bounds, obj_bound):
        self.name = name
        # Should be as xomol.names to maintain order of LOWs and UPs
        names = ["aa", "sced", "cplx", "site", "resl", "acap", "etat", "rely",
                 "Data", "prec", "pmat", "aexp", "flex", "pcon", "tool", "time",
                 "stor", "docu", "b", "plex", "pcap", "kloc", "ltex", "pr",
                 "ruse", "team", "pvol"]
        # Generic Bounds as per menzies.us/pdf/06xomo101.pdf fig.9
        common_bounds = {"aa"  : (1, 6),
                         "sced": (1.00, 1.43),
                         "cplx": (0.73, 1.74),
                         "site": (0.80, 1.22),
                         "resl": (1.41, 7.07),
                         "acap": (0.71, 1.42),
                         "etat": (1, 6),
                         "rely": (0.82, 1.26),
                         "Data": (0.90, 1.28),
                         "prec": (1.24, 6.20),
                         "pmat": (1.56, 7.80),
                         "aexp": (0.81, 1.22),
                         "flex": (1.01, 5.07),
                         "pcon": (0.81, 1.29),
                         "tool": (0.78, 1.17),
                         "time": (1.00, 1.63),
                         "stor": (1.00, 1.46),
                         "docu": (0.81, 1.23),
                         "b"   : (3, 10),
                         "plex": (0.85, 1.19),
                         "pcap": (0.76, 1.34),
                         "kloc": (2, 1000),
                         "ltex": (0.84, 1.20),
                         "pr"  : (1, 6),
                         "ruse": (0.95, 1.24),
                         "team": (1.01, 5.48),
                         "pvol": (0.87, 1.30)}

        self.bound = dict()
        for n in names:
            if n in specific_bounds:
                self.bound[n] = specific_bounds[n]
            else:
                self.bound[n] = common_bounds[n]

        for key, val in self.bound.items():
            if min(val) == max(val):
                self.bound[key] = (min(val), max(val) + 0.000001)  # avoid divide-by-zero error

        self.decNum = len(names)
        self.decs = names
        self.obj_bound = obj_bound
        self.objNum = 4
        self.columns = names + ['o' + str(i) + '_' for i in range(self.objNum)]

        # FOR THE DEAP MODULES, use creator.Ind_xomo as individual type
        if not hasattr(creator, 'F4m'):
            creator.create('F4m', base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))

        if not hasattr(creator, 'Ind_xomo'):
            creator.create('Ind_xomo', array.array, typecode='d', fitness=creator.F4m)

    def _eval(self, df, index, normalized=True):
        dind = []
        for dn in self.decs:
            m, M = self.bound[dn]
            v = df.loc[index, dn]
            dind.append(v * (M - m) + m)

        xomoxo = xomol()
        output = xomoxo.run(dind)
        if not normalized:
            res = output
        else:
            noutput = list()
            for (m, M), v in zip(self.obj_bound, output):
                if v > M:
                    noutput.append(1)
                else:
                    noutput.append((v - m) / (M - m))
            res = noutput

        for i in range(self.objNum):
            df.loc[index, 'o%d_' % i] = round(res[i], 4)

    def init_random_pop(self, size):
        """ return a DataFrame
        Note: all objective were set as -1, an indicator of not assigned.
        :param size: number of population
        :return: pd.DataFrame
        """
        df = pd.DataFrame(data=np.random.rand(size, len(self.columns)), columns=self.columns)

        for i in range(self.objNum):
            df['o%d_' % i] = -1

        return df

    def eval_pd_df(self, df, normalized=True):
        for ind in df.index:
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
            config_obj = creator.Ind_xomo(config)

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


# bounds specific to osp model
bounds_osp = {"prec": (4.96, 6.2),
              "flex": (1.01, 4.05),
              "resl": (4.24, 7.07),
              "team": (3.29, 4.38),
              "pmat": (3.12, 7.8),
              "rely": (1.26, 1.26),
              "cplx": (1.34, 1.74),
              "Data": (1, 1),
              "ruse": (0.95, 1.07),
              "time": (1, 1.63),
              "stor": (1, 1.17),
              "pvol": (0.87, 0.87),
              "acap": (1, 1.19),
              "pcap": (1, 1),
              "pcon": (1, 1.12),
              "aexp": (1, 1.1),
              "plex": (1, 1),
              "ltex": (0.91, 1.09),
              "tool": (1, 1.09),
              "sced": (1, 1.43),
              "site": (1, 1),
              "docu": (0.91, 1.11),
              "kloc": (75, 125)}
bounds_osp2 = {"prec": (1.24, 3.72),
               "flex": (3.04, 3.04),
               "resl": (2.83, 2.83),
               "team": (3.29, 3.29),
               "pmat": (1.56, 3.12),
               "rely": (1.26, 1.26),
               "cplx": (1.34, 1.74),
               "Data": (1.14, 1.14),
               "ruse": (0.95, 1.07),
               "time": (1, 1),
               "stor": (1, 1),
               "pvol": (1, 1),
               "acap": (0.85, 1.19),
               "pcap": (1, 1),
               "pcon": (1, 1.12),
               "aexp": (0.88, 1.1),
               "plex": (0.91, 1),
               "ltex": (0.84, 1.09),
               "tool": (0.78, 1.09),
               "sced": (1, 1.14),
               "site": (0.8, 1),
               "docu": (1, 1.11),
               "kloc": (75, 125)}
bounds_ground = {"prec": (1.24, 6.2),
                 "flex": (1.01, 5.07),
                 "resl": (1.41, 7.07),
                 "team": (1.01, 5.48),
                 "pmat": (1.56, 7.8),
                 "rely": (0.82, 1.1),
                 "cplx": (0.73, 1.17),
                 "Data": (0.9, 1),
                 "ruse": (0.95, 1.24),
                 "time": (1, 1.11),
                 "stor": (1, 1.05),
                 "pvol": (0.87, 1.3),
                 "acap": (0.71, 1),
                 "pcap": (0.76, 1),
                 "pcon": (0.81, 1.29),
                 "aexp": (0.81, 1.1),
                 "plex": (0.91, 1.19),
                 "ltex": (0.91, 1.2),
                 "tool": (1.09, 1.09),
                 "sced": (1, 1.43),
                 "site": (0.8, 1.22),
                 "docu": (0.81, 1.23),
                 "kloc": (11, 392)}
bounds_flight = {"prec": (6.2, 1.24),
                 "flex": (5.07, 1.01),
                 "resl": (7.07, 1.41),
                 "team": (5.48, 1.01),
                 "pmat": (6.24, 4.68),
                 "rely": (1, 1.26),
                 "cplx": (1, 1.74),
                 "Data": (0.9, 1),
                 "ruse": (0.95, 1.24),
                 "time": (1, 1.11),
                 "stor": (1, 1.05),
                 "pvol": (0.87, 1.3),
                 "acap": (1, 0.71),
                 "pcap": (1, 0.76),
                 "pcon": (1.29, 0.81),
                 "aexp": (1.22, 0.81),
                 "plex": (1.19, 0.91),
                 "ltex": (1.2, 0.91),
                 "tool": (1.09, 1.09),
                 "sced": (1, 1),
                 "site": (1.22, 0.8),
                 "docu": (0.81, 1.23)}
# objs_bound_osp = [[0, 1.1e4], [0, 80], [0, 1.5e5], [0, 12]]
# objs_bound_osp2 = [[0, 1.1e4], [0, 80], [0, 1.5e5], [0, 10]]
# objs_bound_groud = [[0, 1.2e4], [0, 80], [0, 1.3e5], [0, 11]]
# objs_bound_flight = [[0, 1.2e4], [0, 80], [0, 1.5e5], [0, 10]]
objs_bound = [[0, 8e3], [0, 65], [0, 1.3e5], [0, 10]]


def get_xomo(version):
    if version == 'osp':
        return XOMO('osp', bounds_osp, objs_bound)
    if version == 'osp2':
        return XOMO('osp2', bounds_osp2, objs_bound)
    if version == 'ground':
        return XOMO('ground', bounds_ground, objs_bound)
    if version == 'flight':
        return XOMO('flight', bounds_flight, objs_bound)


if __name__ == '__main__':
    model = get_xomo('osp')
    df = model.init_random_pop(5)
    model.eval_pd_df(df, normalized=False)
    print(df)
