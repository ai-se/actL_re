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

import glob
import io
import json
import os
import os.path
import time

import numpy as np
import pandas as pd

from Stats.gs import GS
from Stats.hypervolume.pyhv import hypervolume
from Stats.pd_dominance import cull

"""
In file records/raw/results_***.xml

dec_objs
@ json object {model = ..., runtime =.., algorithm=...}
C
O
N
T
E
N
T
# END
"""

"""
In file records/quality_xxx.xml

quality
json object {model=... , algorithm=..., runtime/gs/hv/=...}
} 
"""


def write_results_to_txt(experiment_id, results, model, algorithm_name, runtime=None):
    """
    Writing/APPENDING (if existed) results to a txt file
    :param experiment_id:
    :param results: pandas DataFrame object
    :param model:
    :param algorithm_name:
    :param runtime:
    :return:
    """
    # create the raw as the folder if not existed
    if not os.path.exists('records/raw'):
        os.makedirs('records/raw')

    # create the json object to write in to the file
    info_dict = {'model'  : model if type(model) is str else model.name, 'algorithm': algorithm_name,
                 'runtime': runtime, 'writeAt': time.time()}
    with open('records/raw/' + experiment_id + '.txt', 'a+') as f:
        f.write('@')
        json.dump(info_dict, f)
        f.write('\n')

        # writing the outputs
        f.write(results.to_csv(sep=' '))
        f.write('#\n')


def _read_next_raw(file_name_or_exp_id, folder='records/raw/'):
    """
    reading next raw data or true frontier
    :param file_name_or_exp_id:
    :return: the info - json object.
             the res - pandasDataFrame
    """
    if file_name_or_exp_id.endswith('.txt'):
        file_name_or_exp_id = file_name_or_exp_id[:-4]

    info, res = None, ''
    with open('%s.txt' % (folder + file_name_or_exp_id), 'r') as f:
        for l in f.readlines():
            if not (l.startswith('#') or l.startswith('@')):
                res += l
            elif l.startswith('#'):
                res = pd.read_csv(io.StringIO(res), sep=' ', index_col=0)
                yield info, res
                info, res = None, ''
            elif l.startswith('@'):
                info = json.loads(l[1:])


def report_quality(experiment_id):
    """
    Reporting frontier quality. all results are written to quality/....txt
    Re-calculating everything every time
    :param experiment_id:
    :return:
    """
    # create the quality as the folder if not existed
    if not os.path.exists('records/quality'):
        os.makedirs('records/quality')

    with open('records/quality/%s.txt' % experiment_id, 'w') as f:
        for info, res in _read_next_raw(experiment_id):
            # filter the objectives. make sure the obj is in correct order
            obj_matrix = res.filter(regex='o*_', axis=1)
            obj_matrix = obj_matrix.reindex(sorted(obj_matrix.columns, key=lambda i: int(i[1:-1])), axis=1)

            # METRIC 1 runtime
            json.dump({'model'    : info['model'],
                       'algorithm': info['algorithm'],
                       'runtime'  : info['runtime']
                       }, f)
            f.write('\n')

            # METRIC 2 hv (hypervolume)
            hv = hypervolume(obj_matrix.values.tolist(), np.array([1.0] * obj_matrix.shape[1]))
            json.dump({'model'    : info['model'],
                       'algorithm': info['algorithm'],
                       'hv'       : hv
                       }, f)
            f.write('\n')

            # METRIC3 gs (spread)
            # to calc spread, get the true frontier first
            for ti, tf in _read_next_raw('true.txt', 'records/'):
                if ti['model'] == info['model']:
                    break
            PF0 = tf[obj_matrix.columns].values.tolist()
            PFc = obj_matrix.values.tolist()
            if len(PFc) > 1 and len(PF0) > 1:
                gs = GS(PF0, PFc)
                json.dump({'model'    : info['model'],
                           'algorithm': info['algorithm'],
                           'gs'       : gs
                           }, f)
                f.write('\n')

            # pdb.set_trace()


def construct_PF0(sinceTimeStamp=0):
    """
    All true frontiers are written to file records/true.txt
    :param sinceTimeStamp: UNIX timestamp reference to https://www.unixtimestamp.com/
    :return:
    """
    files = glob.glob("records/raw/*.txt")
    experiment_ids = [i[len('records/raw/'): -len('.txt')] for i in files]
    MTX = dict()

    for exp in experiment_ids:
        for info, results in _read_next_raw(exp):
            model = info['model']
            if info['writeAt'] < sinceTimeStamp: continue
            if model not in MTX:
                MTX[model] = results
            else:
                MTX[model] = pd.merge(MTX[model], results, how='outer')

    tf_file = open('records/true.txt', 'w')
    for m, all_res in MTX.items():
        objs = [i for i in all_res.columns if i.startswith('o') and i.endswith('_')]
        best, dominated = cull(all_res[objs])
        best_res = all_res.loc[best]

        tf_file.write('@')
        json.dump({'model'  : m,
                   'writeAt': time.time()
                   }, tf_file)
        tf_file.write('\n')
        tf_file.write(best_res.to_csv(sep=' '))
        tf_file.write('#\n')

    tf_file.close()


if __name__ == '__main__':
    # construct_PF0()
    report_quality('May08_hpc_100881')
    # for info, res in _read_next_raw('debug_writing2'):
    #     print(info, res)
    #     pdb.set_trace()
