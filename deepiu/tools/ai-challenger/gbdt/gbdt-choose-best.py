#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gbdt-choose-best.py
#        \author   chenghuige  
#          \date   2017-11-24 23:11:51.563739
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

best_score = {}
best_caption = {}

out = open('./ensemble.inference.gbdt_result.best.txt', 'w')

for line in open('./ensemble.inference.gbdt_result.txt'):
  img, caption, score = line.strip().split('\t')
  score = float(score)
  if img not in best_score:
    best_score[img] = score 
    best_caption[img] = caption 
  else:
    if score > best_score[img]:
      best_score[img] = score 
      best_caption[img] = caption 

for img in best_score:
  print(img, best_caption[img], best_score[img], sep='\t', file=out)
  
