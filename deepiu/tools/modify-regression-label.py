#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-label.py
#        \author   chenghuige  
#          \date   2017-11-24 15:52:00.828515
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

out = open('./ensemble.train.modifylabel.txt', 'w')

m = {}
is_header = True
for line in open('./ensemble.train.final.txt'):
  if is_header:
    l = line.strip().split('\t')
    print('\t'.join(l), file=out)
    is_header = False 

  img, caption, score, feature = line.strip().split('\t', 3)
  key = '%s\t%s' % (img, caption)
  m[key] = feature
  
for line in open('./ensemble.best_metrics.txt'):
  l = line.strip().split('\t')
  img, caption, bleu4, cider = l[0], l[1].replace(' ', ''), l[2], l[3]
  
  #score = float(cider)
  score = float(bleu4)
  
  key = '%s\t%s' % (img, caption)
  print(img, caption, score, m[key], sep='\t', file=out)
