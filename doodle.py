#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# No rights reserved

import pdb

import pandas as pd

s = pd.Series(data=[1, 2, 3, 4], index=['a', 'c', 'b', 'd'])
print(s['c'])
pdb.set_trace()
