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


best_score_map = {}
best_caption_map = {}

score_map = {}

infile = './result/0.inst.txt'
if len(sys.argv) > 1:
  infile = sys.argv[1]

infiles = infile.split(',')

for infile in infiles:
  is_header = True
  for line in open(infile):
    if is_header:
      is_header = False
      continue
  
    l = line.split('\t')
    img, caption = l[0].split('|')
    score = float(l[3])
  
    if len(caption) / 3 < 10:
      score -= 10 

    #if '衣着休闲' in caption:
    #  score -= 5

    #if '衣着各异' in caption:
    #  score -= 1
  
    if img not in score_map:
      score_map[img] = {}
    if caption not in score_map[img]:
      score_map[img][caption] = score 
    else:
      score_map[img][caption] += score 

    score = score_map[img][caption]

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

index = infile.split('/')[-1].split('.')[0]

if index != '0':
  out = open('./ensemble.gbdt.evaluate-inference.%s.txt' % index, 'w')
else:
  out = open('./ensemble.gbdt.evaluate-inference.txt', 'w')

bleu4 = 0.
cider = 0.
for img in best_score_map:
  caption = best_caption_map[img]
  bleu4 += bleu4_scores[img][caption]
  cider += cider_scores[img][caption]
  print(img, caption, best_score_map[img], sep='\t', file=out)

num_imgs = len(best_score_map)
bleu4 /= num_imgs
cider /= num_imgs

print('bleu4:{} cider:{}'.format(bleu4, cider))

