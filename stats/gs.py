#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2016, Jianfeng Chen <jchen37@ncsu.edu>
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


from __future__ import division


def dist(a, b):
    return sum((i - j) ** 2 for i, j in zip(a, b))


def GS(PF0, PFc):
    # https: // ieeexplore.ieee.org / stamp / stamp.jsp?tp = & arnumber = 996017
    # sorting the objectives
    PF0 = sorted(PF0, key=lambda ind: [o for o in ind])
    PFc = sorted(PFc, key=lambda ind: [o for o in ind])

    df = dist(PF0[0], PFc[0])
    dl = dist(PF0[-1], PFc[-1])
    di = [dist(i, j) for i, j in zip(PFc[:-1], PFc[1:])]
    d_avg = sum(di) / len(di)

    dr = sum([abs(i - d_avg) for i in di])
    gs = (df + dl + dr) / (df + dl + (len(di) - 1) * d_avg)
    return gs
