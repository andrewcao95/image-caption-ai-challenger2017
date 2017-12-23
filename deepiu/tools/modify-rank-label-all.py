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

#def get_cider_rank(cider):
#  score = 0 
#  if cider > 4.5:
#    score = 4
#  elif cider > 3.:
#    score = 3
#  elif cider > 2.:
#    score = 2 
#  elif cider > 1.:
#    score = 1
#  return score 

def get_cider_rank(cider):
  score = 0 
  if cider > 4.5:
    score = 6
  elif cider > 3.:
    score = 5
  elif cider > 2.:
    score = 4 
  elif cider > 1.:
    score = 3
  elif cider > 0.5:
    score = 2
  elif cider > 0.2:
    score = 1
  return score 

#def get_bleu4_rank(bleu4):
#  score = 0 
#  if bleu4 > 0.9:
#    score = 4 
#  elif bleu4 > 0.8:
#    score = 3
#  elif bleu4 > 0.7:
#    score = 2
#  elif bleu4 > 0.5:
#    score = 1
#  return score

def get_bleu4_rank(bleu4):
  score = 0 
  if bleu4 > 0.95:
    score = 6
  if bleu4 > 0.9:
    score = 5 
  elif bleu4 > 0.8:
    score = 4
  elif bleu4 > 0.7:
    score = 3
  elif bleu4 > 0.5:
    score = 2
  elif bleu4 > 0.1:
    score = 1
  return score

def get_meteor_rank(metric):
  score = 0 
  if metric > 0.95:
    score = 5
  elif metric > 0.6:
    score = 4
  elif metric > 0.5:
    score = 3
  elif metric > 0.4:
    score = 2
  elif metric > 0.3:
    score = 1
  return score

def get_mix_rank(cider, bleu4):
  score = 0
  if bleu4 > 0.95 and cider > 5.0:
    score = 8
  elif bleu4 > 0.9 and cider > 4.5:
    score = 7
  elif bleu4 > 0.9 and cider > 3:
    score = 6
  elif bleu4 > 0.8 and cider > 3:
    score = 5
  elif bleu4 > 0.7 and cider > 2:
    score = 4
  elif cider > 1.2:
    score = 3
  elif cider > 0.5:
    score = 2
  elif cider > 0.1:
    score = 1
  if bleu4 < 0.1:
    score = max(0, score - 2)
  return score

#def get_mix_rank(cider, bleu4):
#  score = 0
#  if bleu4 > 0.9 and cider > 3.:
#    score = 9
#  elif bleu4 > 0.8 and cider > 2.0:
#    score = 8
#  elif bleu4 > 0.7 and cider > 1.4:
#    score = 7  
#  elif bleu4 > 0.6 and cider > 1.0:
#    score = 6
#  elif bleu4 > 0.5 and cider > 0.8:
#    score = 5
#  elif bleu4 > 0.4 and cider > 0.5:
#    score = 4
#  elif bleu4 > 0.3 and cider > 0.4:
#    score = 3
#  elif bleu4 > 0.2 and cider > 0.3:
#    score = 2
#  elif bleu4 > 0.0001 and cider > 0.0001:
#    score = 1
#  return score

label_type = sys.argv[1]

ofile = './ensemble.train.modifylabel.txt'
if len(sys.argv) > 2:
  ofile = sys.argv[2]

out = open(ofile, 'w')

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
  
for line in open('./ensemble.best_metrics_all.txt'):
  l = line.strip().split('\t')
  img, caption, bleu4, cider, meteor, rouge = l[0], l[1].replace(' ', ''), l[2], l[3], l[4], l[5]
  
  cider = float(cider)
  #score = get_cider_rank(cider)

  # bleu4 is better then cider 
  bleu4 = float(bleu4)
  #score = get_bleu4_rank(bleu4) 

  meteor = float(meteor)
  rouge = float(rouge)

  if label_type == 'cider':
    score = get_cider_rank(cider)
  elif label_type == 'rcider':
    score = cider
  elif label_type == 'bleu4':
    score = get_bleu4_rank(bleu4)
  elif label_type == 'rbleu4':
    score = bleu4
  elif label_type == 'meteor':
    score = get_meteor_rank(meteor)
  elif label_type == 'rmeteor':
    score = meteor
  elif label_type == 'rrouge':
    score = rouge
  else:
    score = get_mix_rank(cider, bleu4)

  #score = cider
  
  key = '%s\t%s' % (img, caption)
  print(img, caption, score, m[key], sep='\t', file=out)
