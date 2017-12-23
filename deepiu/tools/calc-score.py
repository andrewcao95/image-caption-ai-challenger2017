#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calc-score.py
#        \author   chenghuige  
#          \date   2017-11-24 16:24:02.248521
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import melt 
logging = melt.logging
import gezi

is_header = True

best_score_map = {}
best_caption_map = {}

for line in open('./result/0.inst.txt'):
  if is_header:
    is_header = False
    continue

  l = line.split('\t')
  img, caption = l[0].split('|')
  score = float(l[2])

  if img not in best_score_map:
    best_score_map[img] = score
    best_caption_map[img] = caption
  else:
    if score > best_score_map[img]:
      best_score_map[img] = score 
      best_caption_map[img] = caption 

timer = gezi.Timer('load bleu4 and cider info')
bleu4_scores = {}
cider_scores = {}
for line in open('./ensemble.best_metrics.txt'):
  l = line.split('\t')
  img, caption, bleu4, cider = l[0], l[1], l[2], l[3]
  bleu4_scores.setdefault(img, {})
  bleu4_scores[img][caption] = float(bleu4)
  cider_scores.setdefault(img, {})
  cider_scores[img][caption] = float(cider)
timer.print()

out = open('./ensemble.gbdt.evaluate-inference.txt', 'w')
bleu4 = 0
cider = 0
for img in best_score_map:
  caption = best_caption_map[img]
  bleu4 += bleu4_scores[img][caption]
  cider += cider_scores[img][caption]
  print(img, caption, best_score_map[img], sep='\t', file=out)

num_imgs = len(best_score_map)
bleu4 /= num_imgs
cider /= num_imgs

print('bleu4:{} cider:{}'.format(bleu4, cider))

