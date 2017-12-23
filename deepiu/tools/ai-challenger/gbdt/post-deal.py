#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   post-deal.py
#        \author   chenghuige  
#          \date   2017-11-27 10:16:37.698571
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

ifile = sys.argv[1]
ofile = ifile.replace('.txt', '.postdeal.txt')
with open(ofile, 'w') as out:
  for line in open(ifile):
    line = line.strip()
    line = line.replace('商店', '超市')
    print(line, file=out)
