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
  if weights == 'None':
    weights = [1.0] * len(files)
  else:
    weights = map(float, weights.split(','))
else:
  weights = [1.0] * len(files)

# experiments show logits is better then prob for enseemble so 
method = 'logits'
if len(sys.argv) > 4:
  method = sys.argv[4]

score_col = None
if method == 'logits':
  score_col = -2
elif method == 'top_predictions':
  score_col = 3
elif method == 'top_logits':
  score_col = 2
elif method == 'predictions':
  score_col = -1

print('method', method, 'score_col', score_col, file=sys.stderr)
assert score_col

dir = os.path.dirname(files[0])

ofile = os.path.join(dir, 'ensemble.%s.txt'%type)
out = open(ofile, 'w')

print('files:', files, 'len(files)', len(files), file=sys.stderr)
print('weights:', weights, file=sys.stderr)
print('ofile:', ofile, file=sys.stderr)

m = {}
use_log = False 

infos = {}

candidates = {}
for file in files:
  for line in open(file):
    l = line.strip().split('\t')
    img, top_classes = l[0], l[1]
    if img not in candidates:
      candidates[img] = set(top_classes.split())
    else:
      for item in top_classes.split():
        candidates[img].add(item)

for file, weight in zip(files, weights):
  for line in open(file):
    l = line.strip().split('\t')
    img, top_classes, scores = l[0], l[1], l[score_col]
    
    predictions = map(float, l[-1].split()) 
    predictions = [x * weight for x in predictions]
    l[-1] = predictions
    if img not in infos:
      infos[img] = l 
    else:
      pre_predictions = infos[img][-1]
      for i in range(len(predictions)):
        predictions[i] = pre_predictions[i] + predictions[i]
      infos[img][-1] = predictions 

    if img not in m:
      m[img] = {}

    top_classes = top_classes.split(' ')
    scores = map(float, scores.split(' '))
    if len(scores) == len(top_classes):
      for top_class, score in zip(top_classes, scores):
        m[img].setdefault(top_class, 0.)
        if use_log:
          score = math.log(score)
        m[img][top_class] += score * weight 
    else:
      for top_class in candidates[img]:
        m[img].setdefault(top_class, 0.)
        score = scores[int(top_class)]
        if use_log:
          score = math.log(score)
        m[img][top_class] += score * weight

for img, result in m.items():
  if not use_log:
    l = [(x, y) for x, y in result.items()]
  else:
    l = [(x, math.exp(y)) for x, y in result.items()]
  sorted_result = sorted(l, key=operator.itemgetter(1), reverse=True)

  #print('-------------', sorted_result)

  top_classes, scores = zip(*sorted_result)
  scores = [x / len(files) for x in scores]
  top_classes = ' '.join(top_classes)
  scores = ' '.join(map(str, scores))

  info = infos[img]

  #print('-----', info)
  info[1] = top_classes 
  if 'prediction' in method:
    info[3] = scores
  else:
    info[2] = scores
  info[-1] = ' '.join(map(str, [x / len(files) for x in info[-1]]))

  print('\t'.join(info), file=out)

