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
import os.path
import pdb
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import toolz
from deap import creator
from deap.tools import sortLogNondominated
from lxml import etree

from Benchmarks.POM3 import get_pom3
from stats.hypervolume.pyhv import hypervolume

"""
In file records/raw_results_***.xml

dec_objs
|
|-- model M
|-- |-- algorithm A
|-- |-- | -- repeat
|-- |-- | -- | -- c (decs =... objs = ...)
|-- |-- | -- | -- c (decs =... objs = ...)
|-- |-- | -- | -- c (decs =... objs = ...)
|-- |-- | -- | -- c (decs =... objs = ...)
"""

"""
In file records/quality_xxx.xml

quality
|
|-- model M
|-- |-- algorithm A
|-- |-- | -- r (hv=..., gs=...)
|-- |-- | -- r (hv=..., gs=...)
|-- |-- | -- r (hv=..., gs=...)
|-- |-- | -- r (hv=..., gs=...)
"""


def _list2_str(lst):
    return " ".join(str(round(x, 3)) for x in lst)


def _str2list(s):
    # '0.292 0.587 0.09' -> [0.292, 0.587, 0.09]
    return [float(i) for i in s.split(' ')]


def write_results_to_xml(experiment_id, results, model, algorithm_name, runtime=None):
    """
    Writing/APPENDING (if existed) results to a xml file
    :param experiment_id:
    :param results:
    :param model:
    :param algorithm_name:
    :param runtime:
    :return:
    """
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

    runtime = runtime or -1
    thisrepeat = ET.SubElement(algNode, 'repeat', timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                               runtime=str(runtime))

    for i in results:
        ET.SubElement(thisrepeat, 'c', decision=_list2_str(i.tolist()), objectives=_list2_str(i.fitness.values))
    # writing out
    tree.write(file)


def _read_xml_to_obj_list(experiment_id, model, algorithm_name):
    file = "records/raw_results_%s.xml" % experiment_id
    tree = ET.parse(file)
    root = tree.getroot()
    mnode = None
    for x in root.findall('model'):
        if x.attrib['name'] == model.name:
            mnode = x
            break
    if not mnode: return list()

    algNode = None
    for x in mnode.findall('algorithm'):
        if x.attrib['name'] == algorithm_name:
            algNode = x
            break
    if not algNode: return list()

    RES = list()
    for repeat in algNode.findall('repeat'):
        r = list()
        for c in repeat.findall('c'):
            r.append(_str2list(c.attrib['objectives']))
        RES.append(r)
    return RES


def report_quality(experiment_id, models, algorithm_names):
    """
    Reporting frontier quality
    :param experiment_id:
    :param models:
    :param algorithm_names:
    :return:
    """
    root = ET.Element('quality')

    for model in models:
        mnode = ET.SubElement(root, 'model', name=model.name, decNum=str(model.decNum), objNum=str(model.objNum))
        for algorithm in algorithm_names:
            algNode = ET.SubElement(mnode, 'algorithm', name=algorithm)
            res = _read_xml_to_obj_list(experiment_id, model, algorithm)
            for repeat in res:
                hv = hypervolume(repeat, np.array([1.0] * model.objNum))
                pdb.set_trace()
                ET.SubElement(algNode, 'r', hv=str(round(hv, 3)), gs='1', runtime='1')

    file = "records/quality_%s.xml" % experiment_id
    tree = ET.ElementTree(root)
    tree.write(file)


def construct_PF0_xml(sinceTimeStamp=0):
    """
    :param sinceTimeStamp: UNIX timestamp reference to https://www.unixtimestamp.com/
    :return:
    """
    frontiers = dict()

    files = glob.glob("records/raw_results_*.xml")
    for file in files:  # for each experiment
        root = ET.parse(file).getroot()
        for mnode in root.findall('model'):  # for each model
            name = mnode.attrib['name']
            if name not in frontiers:
                frontiers[name] = list()
            for algNode in mnode.findall('algorithm'):  # for each algorithm
                for repeatNode in algNode.findall('repeat'):  # for each repeat
                    timeat = datetime.strptime(repeatNode.attrib['timestamp'], "%Y-%m-%d %H:%M:%S")
                    if timeat < datetime.fromtimestamp(sinceTimeStamp): continue
                    for cNode in repeatNode.findall('c'):  # for every got frontier points
                        frontiers[name].append(cNode)

    troot = ET.Element('TrueFrontier')

    # retrieving the "TRUE" frontiers
    creator.create('xmlInd', str, fitness=creator.FitnessMin)
    for model in frontiers.keys():
        mnode = ET.SubElement(troot, 'model', name=model)

        pops = list()
        for c in frontiers[model]:
            ind = creator.xmlInd(c.attrib['decision'])
            ind.fitness.values = _str2list(c.attrib['objectives'])
            pops.append(ind)
        best = sortLogNondominated(pops, k=10, first_front_only=True)  # k value is non-sense

        for b in best:
            ET.SubElement(mnode, 'c', decision=b, objectives=_list2_str(b.fitness.values))

    tree = ET.ElementTree(troot)
    tree.write("records/true_frontier.xml")


if __name__ == '__main__':
    model = get_pom3('p3a')
    # report_quality('debug_writing', [get_pom3('p3a'), get_pom3('p3b')], ['riot', 'nsgaii'])
    construct_PF0_xml()
