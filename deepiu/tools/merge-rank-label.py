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

def get_cider_rank(cider):
  score = 0 
  if cider > 3.:
    score = 4
  elif cider > 2.2:
    score = 3
  elif cider > 1.4:
    score = 2 
  elif cider > 0.7:
    score = 1
  return score 

def get_bleu4_rank(bleu4):
  score = 0 
  if bleu4 > 0.9:
    score = 4 
  elif bleu4 > 0.8:
    score = 3
  elif bleu4 > 0.7:
    score = 2
  elif bleu4 > 0.5:
    score = 1
  return score

def get_mix_rank(cider, bleu4):
  score = 0
  if bleu4 > 0.9 and cider > 3.:
    score = 6
  elif bleu4 > 0.8 and cider > 2.2:
    score = 5
  elif bleu4 > 0.7 and cider > 1.4:
    score = 4  
  elif bleu4 > 0.5 and cider > 0.7:
    score = 3
  elif bleu4 > 0.4 and cider > 0.5:
    score = 2 
  elif bleu4 > 0.3 and cider > 0.4:
    score = 1
  return score

out = open('./ensemble.train.txt', 'w')

m = {}
is_header = True
for line in open('./ensemble.evaluate-inference.feature.txt'):
  if is_header:
    l = line.strip().split('\t')
    l.insert(2, 'score')
    print('\t'.join(l), file=out)
    is_header = False 

  img, caption, feature = line.strip().split('\t', 2)
  key = '%s\t%s' % (img, caption)
  m[key] = feature
  
for line in open('./ensemble.best_metrics.txt'):
  l = line.strip().split('\t')
  img, caption, bleu4, cider = l[0], l[1].replace(' ', ''), l[2], l[3]
  
  cider = float(cider)
  #score = get_cider_rank(cider)
  # bleu4 is better then cider 
  bleu4 = float(bleu4)
  
  #score = get_bleu4_rank(bleu4)

  score = get_mix_rank(cider, bleu4)
  
  key = '%s\t%s' % (img, caption)
  print(img, caption, score, m[key], sep='\t', file=out)
  #print(img, caption, score)
