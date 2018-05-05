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

import pdb

import numpy as np
import pandas as pd

"""
Get the Pareto Frontier of a Pandas DataFrame
https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
Not an efficient algorithm, but effective.
"""


def _dominates(row, rowCandidate):
    """
    All objectives are less the better
    First dominates the second one
    """
    return all(r <= rc for r, rc in zip(row.values, rowCandidate.values))


def cull(pts):
    """
    :param pts: pd.DataFrame
    :return: cleared. dominated. the index
    """
    dominated = []
    cleared = []
    remaining = pts.index

    while remaining.size:
        candidate = remaining[0]
        for other in remaining[1:]:
            if _dominates(pts.loc[candidate], pts.loc[other]):
                dominated.append(other)
                remaining = remaining.drop(other)

        remaining = remaining.drop(candidate)
        if not any(_dominates(pts.loc[other], pts.loc[candidate]) for other in remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)

    return cleared, dominated


if __name__ == '__main__':
    # pts = [[1, 3, 6], [0, 7, 8], [2, 3, 7], [6, 1, 4], [4, 4, 1], [0, 2, 6]]
    pts = np.random.rand(10, 3)
    df = pd.DataFrame(pts, columns=['o0', 'o1', 'o2'])
    cleared, dominated = cull(df)
    pdb.set_trace()
