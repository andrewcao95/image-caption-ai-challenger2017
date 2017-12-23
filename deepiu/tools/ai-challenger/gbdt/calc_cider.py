#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calc_cider.py
#        \author   chenghuige  
#          \date   2017-11-21 19:48:35.770886
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

cider = 0
for i, line in enumerate(open('./ensemble.caption_metrics.txt')):
  if i == 0:
    continue
  l = line.split('\t')
  cider += float(l[8])
  if i == 30000:
    break

print(cider / 30000)
