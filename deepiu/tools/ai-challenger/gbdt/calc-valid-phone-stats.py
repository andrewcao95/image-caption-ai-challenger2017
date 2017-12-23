#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calc-valid-hand-stats.py
#        \author   chenghuige  
#          \date   2017-11-30 00:33:15.133092
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

count = 0
for line in open('./ensemble.gbdt.caption_metrics.txt'):
  l = line.strip().split('\t')
  label_captions = l[2].split('|')
  caption = label_captions[0]
  if '手机' in caption:
    count += 1

print(count)

