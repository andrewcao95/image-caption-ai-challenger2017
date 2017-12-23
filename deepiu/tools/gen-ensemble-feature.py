#!/usr/bin/env python
#encoding=utf8
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
import cPickle as pickle

import melt 
import gezi
import jieba

from deepiu.util import idf
from deepiu.util import text2ids

text2ids.init('/home/gezi/mine/mount/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt')

input = sys.argv[1]
type = sys.argv[2]

if type.startswith('eval'):
  type = 'evaluate-inference'

if ',' in input:
  files = input.split(',')
else:
  files = glob.glob(input + '/*model*.%s.txt'%type)
  if not 'ensemble' in type:
    files = [x for x in files if not 'ensemble' in x]

files.sort(key=os.path.getmtime)

if len(sys.argv) > 3:
  weights = sys.argv[3]
  weights = map(float, weights.split(','))
else:
  weights =[1.0] * len(files)

if len(weights) < len(files):
  weights = [1.] * (len(files) - len(weights)) + weights

dir = os.path.dirname(files[0])

ofile = os.path.join(dir, 'ensemble.%s.feature.txt'%type)
out = open(ofile, 'w')

logfile = open('./log.txt', 'w')
print(weights, file=logfile)

print('files:', files, 'len(files)', len(files), file=sys.stderr)
#print('weights:', weights, file=sys.stderr)
for file_, weight in zip(files, weights):
  print(weight, file_, file=sys.stderr)
  print(weight, file_, file=logfile)

print('ofile:', ofile, file=sys.stderr)

def get_num_words(l):
  for i in range(len(l)):
      if l[i] == 0:
        return i 
  return len(l)

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

m = {}
ensemble_map = {}

# TODO add tfidf, caption attention info like coverage and other infos, detection same score

cols = ['#image', 'caption']
features = []
features += ['final_ensemble_score', 'ori_ensemble_score', 'mil_score', 'num_votes']
image_caption_feature_len = len(features)
caption_features = ['caption_len', 'caption_words', \
                    'caption_total_idf', 'caption_mean_idf', \
                    'caption_dup_count', 'caption_dup_ratio']
caption_features_len = len(caption_features)
features += caption_features
idx = len(features)
features += [x.strip('./').replace('.inference.txt', '').replace('.evaluate-inference.txt', '') for x in files] 
cols += features

idf_weights = idf.get_idf()

segged_caption = {}

if os.path.exists('segged_caption.pkl'):
  with open('segged_caption.pkl', 'rb') as handle:
    segged_caption = pickle.load(handle)

print('len(segged_caption)', len(segged_caption), file=sys.stderr)

num_segged_caption = len(segged_caption)

for i, (file, weight) in enumerate(zip(files, weights)):
  print(i, file)
  for line in open(file):
    l = line.strip().split('\t')
    img, texts , scores = l[0], l[-2], l[-1]

    #if img != '51b63e3e907db6514763a666a8adbdbb0a9e5661':
    #  continue

    if img not in m:
      m[img] = {}

    texts = texts.split(' ')
    scores = map(float, scores.split(' '))
    for text, score in zip(texts, scores):
      is_first_text = True
      if text in m[img]:
        is_first_text = False
      m[img].setdefault(text, [0.] * len(features))
      m[img][text][idx + i] = score 
      m[img][text][0] += score * weight
      if weight >= 1.0:
        m[img][text][1] += score 
      if weight < 1.0:
        m[img][text][2] += score
      if weight == 1.0:  
        m[img][text][3] += 1.

      # per text feature do not need to calc again
      # TODO still has dup calc for text info
      if is_first_text:
        m[img][text][4] = len(gezi.get_single_cns(text))
        
        if text in segged_caption:
          words = segged_caption[text]
        else:
          words = [x.encode('utf-8') for x in jieba.cut(text)]
          segged_caption[text] = words
        ids = text2ids.words2ids(words)
        num_words = get_num_words(ids)
        m[img][text][5] = num_words

        weights = [idf_weights[x] for x in ids]
        total_idf = sum(weights)
        m[img][text][6] = total_idf
        mean_idf = total_idf / num_words
        m[img][text][7] = mean_idf

        dup, total = cacl_duplicate(words)
        ratio = (total - dup) / float(total)
        m[img][text][8] = dup
        m[img][text][9] = ratio 
    


print('\t'.join(cols), file=out)
for img, captions in m.items():
  for caption, feature in captions.items():
    print(img, caption, '\t'.join(map(str, feature)), sep='\t', file=out)

if len(segged_caption) > num_segged_caption:
  print('now segged_caption with %s' % len(segged_caption))
  with open('segged_caption.pkl', 'wb') as handle:
      pickle.dump(segged_caption, handle, protocol=pickle.HIGHEST_PROTOCOL)
