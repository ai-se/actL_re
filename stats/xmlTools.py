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

import os.path
import time
import xml.etree.ElementTree as ET

import toolz
from lxml import etree

"""
In file records/***.xml

dec_objs
|
|-- model M
|-- |-- algorithm A
|-- |-- | -- repeat
|-- |-- | -- | -- frontier0 (decs =... objs = ...)
|-- |-- | -- | -- frontier2 (decs =... objs = ...)
|-- |-- | -- | -- frontier3 (decs =... objs = ...)
|-- |-- | -- | -- frontier4 (decs =... objs = ...)
"""


def _list_to_str(lst):
    return " ".join(str(round(x, 3)) for x in lst)


def write_results_to_xml(experiment_id, results, model, algorithm_name):
    # removing duplicated res. Individual with the same objective are treated as duplicates
    uni = toolz.unique(results, lambda x: x.fitness.values)
    results = [i for i in uni]

    file = "records/raw_results_%s.xml" % experiment_id
    if not os.path.isfile(file):
        reports = etree.Element('dec_objs')
        with open(file, 'wb') as f:
            f.write(etree.tostring(reports, pretty_print=True))

    tree = ET.parse(file)
    root = tree.getroot()
    mnode = None
    for x in root.findall('model'):
        if x.attrib['name'] == model.name:
            mnode = x
            break

    if not mnode:
        mnode = ET.SubElement(root, 'model', name=model.name, decNum=str(model.decNum), objNum=str(model.objNum))

    algNode = None
    for x in mnode.findall('algorithm'):
        if x.attrib['name'] == algorithm_name:
            algNode = x
            break
    if not algNode:
        algNode = ET.SubElement(mnode, 'algorithm', name=algorithm_name)

    thisrepeat = ET.SubElement(algNode, 'repeat', timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    for i in results:
        ET.SubElement(thisrepeat, 'c', decision=_list_to_str(i.tolist()), objectives=_list_to_str(i.fitness.values))
    # writing out
    tree.write(file)


