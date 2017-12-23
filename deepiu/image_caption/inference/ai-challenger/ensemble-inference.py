#!/usr/bin/env python
# ==============================================================================
#          \file   ensemble-inference.py
#        \author   chenghuige  
#          \date   2017-10-06 21:29:49.558024
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import glob
import operator
import math

input = sys.argv[1]
type = sys.argv[2]

if ',' in input:
  files = input.split(',')
else:
  files = glob.glob(input + '/*model*.%s.txt'%type)
  if not 'ensemble' in type:
    files = [x for x in files if not 'ensemble' in x]

if len(sys.argv) > 3:
  weights = sys.argv[3]
  weights = map(float, weights.split(',')) 
else:
  weights =[1.0] * len(files)

if len(weights) < len(files):
  weights = [1.0] * (len(files) - len(weights)) + weights

dir = os.path.dirname(files[0])

ofile = os.path.join(dir, 'ensemble.%s.txt'%type)
out = open(ofile, 'w')

logfile = open('./log.txt', 'w')
print(weights, file=logfile)

print('files:', files, 'len(files)', len(files), file=sys.stderr)
#print('weights:', weights, file=sys.stderr)
for file_, weight in zip(files, weights):
  print(weight, file_, file=sys.stderr)
  print(weight, file_, file=logfile)

print('ofile:', ofile, file=sys.stderr)

m = {}
use_log = False 
#use_log = True

#smooth = False
#smooth = True 

min_score = {}

#file_set = set() 
#img2texts = {}

#MAX_TEXTS = 10
for file, weight in zip(files, weights):
  #file_set.add(file)  
  #min_score[file] = {}
  for line in open(file):
    l = line.strip().split('\t')
    img, texts , scores = l[0], l[-2], l[-1]
    if img not in m:
      m[img] = {}
      #img2texts[img] = set()
    #min_score[file][img] = 1.0
    texts = texts.split(' ')
    scores = map(float, scores.split(' '))

    #texts = texts[:MAX_TEXTS]
    #scores = scores[:MAX_TEXTS]

    for text, score in zip(texts, scores):
      m[img].setdefault(text, 0.)
      if use_log:
        score = math.log(score)
      m[img][text] += score * weight 
      #img2texts[img].add(text)

      #if score < min_score[file][img]:
      #  min_score[file][img] = score

# if smooth:
#   for file, weight in zip(files, weights):
#     for line in open(file):
#       l = line.strip().split('\t')
#       img, texts  = l[0], l[-2] 
#       texts_set = set(texts.split(' ')) 
#       for text in img2texts[img]:
#         if text not in texts_set:
#           m[img][text] += min_score[file][img] * 0.05 * weight


for img, result in m.items():
  if not use_log:
    l = [(x, y) for x, y in result.items()]
  else:
    l = [(x, math.exp(y)) for x, y in result.items()]
  sorted_result = sorted(l, key=operator.itemgetter(1), reverse=True)
  texts = []
  scores = []
  for text, score in sorted_result:
    texts.append(text)
    scores.append(str(score))
  texts = ' '.join(texts)
  scores = ' '.join(scores)
  print(img, sorted_result[0][0], sorted_result[0][1], texts, scores, sep='\t', file=out)


