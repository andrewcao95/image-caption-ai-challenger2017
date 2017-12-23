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


binfile = sys.argv[1]

ofile = './ensemble.train.modifylabel.txt'
if len(sys.argv) > 2:
  ofile = sys.argv[2]

out = open(ofile, 'w')

num_bins = int(sys.argv[3])

infile = './ensemble.train.final.txt'

m = {}
is_header = True
for line in open(infile):
  if is_header:
    l = line.strip().split('\t')
    print('\t'.join(l), file=out)
    is_header = False 

  img, caption, score, feature = line.strip().split('\t', 3)
  key = '%s\t%s' % (img, caption)
  m[key] = feature
 

for i, line in enumerate(open(binfile)):
  if i == 0:
    continue
  l = line.strip().split('\t')
  img, caption, score = l[0], l[1].replace(' ', ''), l[2] 
  
  score = int(float(score) * num_bins)
  
  key = '%s\t%s' % (img, caption)
  print(img, caption, score, m[key], sep='\t', file=out)
