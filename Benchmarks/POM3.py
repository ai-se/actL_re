from __future__ import division

import numpy as np
import pandas as pd

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
        self.obj_bound = obj_bound

        self.columns = names + ['o' + str(i) + '_' for i in range(self.objNum)]

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
                if v > M:
                    noutput.append(1)
                else:
                    noutput.append((v - m) / (M - m))
            res = noutput

        for i in range(self.objNum):
            df.loc[index, 'o%d_' % i] = res[i]

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


# bounds specific to pom3 model
bounds_pom3a = [[0.1, 0.82, 2, 0.40, 1, 1, 0, 0, 1], [0.9, 1.20, 10, 0.70, 100, 50, 4, 5, 44]]
bounds_pom3b = [[0.10, 0.82, 80, 0.40, 0, 1, 0, 0, 1], [0.90, 1.26, 95, 0.70, 100, 50, 2, 5, 20]]
bounds_pom3c = [[0.50, 0.82, 2, 0.20, 0, 40, 2, 0, 20], [0.90, 1.26, 8, 0.50, 50, 50, 4, 5, 44]]
bounds_pom3d = [[0.10, 0.82, 2, 0.60, 80, 1, 0, 0, 10], [0.20, 1.26, 8, 0.95, 100, 10, 2, 5, 20]]

objs_bound = [[0, 1300], [0, 0.7], [0, 0.65]]


def get_pom3(version):
    if version == 'p3a':
        return POM3('p3a', bounds_pom3a, objs_bound)
    if version == 'p3b':
        return POM3('p3b', bounds_pom3b, objs_bound)
    if version == 'p3c':
        return POM3('p3c', bounds_pom3c, objs_bound)
    if version == 'p3d':
        return POM3('p3d', bounds_pom3d, objs_bound)


if __name__ == '__main__':
    model = get_pom3('p3a')
    df = model.init_random_pop(5)
    model.eval_pd_df(df, normalized=False)
    print(df)
