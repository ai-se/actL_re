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

import json

import matplotlib.pyplot as plt


def box_plot_quality(expId_lst, saving=False):
    if type(expId_lst) is not list:
        expId_lst = [expId_lst]

    all_records = list()
    for expId in expId_lst:
        with open('records/quality/%s.txt' % expId, 'r') as f:
            for l in f.readlines():
                info = json.loads(l)
                all_records.append(info)

    models = set([i['model'] for i in all_records])
    algs = sorted(set([i['algorithm'] for i in all_records]))

    fig, axes = plt.subplots(nrows=max(2, len(models)), ncols=3, figsize=(6, 2 * max(2, len(models))), sharey=False)

    fs = 9  # font size in title
    xpo = [1, 1.5]  # adjusting the x-gap of boxes
    bw = 0.4  # width of boxes

    for mi, model in enumerate(models):
        hv = list()
        gs = list()
        runtime = list()
        for a in algs:
            hv.append([i['hv'] for i in all_records if i['model'] == model and 'hv' in i and i['algorithm'] == a])
            gs.append([i['gs'] for i in all_records if i['model'] == model and 'gs' in i and i['algorithm'] == a])
            runtime.append(
                [i['runtime'] for i in all_records if i['model'] == model and 'runtime' in i and i['algorithm'] == a])

        axes[mi, 0].boxplot(hv, labels=algs, widths=bw, positions=xpo)
        axes[mi, 0].set_title("Hypervolume (to max)", fontsize=fs)
        axes[mi, 0].set_ylabel(model, fontsize=fs * 1.1)

        axes[mi, 1].boxplot(gs, labels=algs, widths=bw, positions=xpo)
        axes[mi, 1].set_title("Spread (to min)", fontsize=fs)

        axes[mi, 2].boxplot(runtime, labels=algs, widths=bw, positions=xpo)
        axes[mi, 2].set_title("Runtime (to min)", fontsize=fs)

    for ax in axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        # ax.set_aspect(1)
        pass

    plt.tight_layout()
    if not saving:
        plt.show()
    else:
        plt.savefig('records/quality/boxplot_%s.pdf' % expId_lst)


if __name__ == '__main__':
    print("this is demonstration for box-plotting.")
    box_plot_quality('May08_hpc_100881', saving=True)
