#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   post-deal.py
#        \author   chenghuige  
#          \date   2017-10-08 16:06:55.933187
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import jieba

import gezi

infile = sys.argv[1]
ofile = sys.argv[2]


def cacl_duplicate(caption):
  m = {}
  num_duplicates = 0
  total = 0
  for w in caption:
    if not gezi.is_single_cn(w):
      if w is '东西' or w is '各异':
        num_duplicates += 2
        total += 2
      if w not in m:
        m[w] = 1
      else:
        m[w] += 1
        if w == '一个' and m[w] < 3:
          continue 
        if w == '戴着' or w == '眼镜' or w == '手机' or w == '帽子':
          total += 2
          num_duplicates += 2
        num_duplicates += 1
      total += 1
  
  if total == 0:
    total = 1
    num_duplicates = 0

  return num_duplicates, total

sep = ' '
out = open(ofile, 'w')
diff_out = open('/tmp/diff.txt', 'w')

for line in open(infile):
  l = line.strip().split('\t')
  img, best_caption, captions, scores = l[0], l[1], l[-2], l[-1]
  scores = map(float, scores.split(sep))
  captions = captions.split(sep)

  for i in range(len(captions)):
    captions[i] = [x.encode('utf-8') for x in jieba.cut(captions[i])]

  results = []
  for caption, score in zip(captions, scores):
    num_duplicates, total = cacl_duplicate(caption)

    ratio = (total - num_duplicates) / float(total)
    if len(caption) < 6:
      ratio *= 0.1
    #ratio = 1
    #print('|'.join(caption), total, num_duplicates,  ratio)
    score *= ratio

    results.append([score, caption])

  results.sort(reverse=True)

  scores, captions = list(zip(*results))

  scores = map(str, scores)
  captions = [''.join(x) for x in captions]
  
  print(img, captions[0],  sep.join(captions), sep.join(scores), sep='\t', file=out)

  if best_caption != captions[0]:
    print(img, best_caption, captions[0], file=diff_out)
