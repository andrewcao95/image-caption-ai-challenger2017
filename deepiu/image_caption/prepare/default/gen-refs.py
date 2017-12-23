#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   gen-refs.py
#        \author   chenghuige  
#          \date   2017-09-24 01:33:55.850122
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
try:
  import cPickle as pickle
except ImportError:
  import pickle

from gezi.metrics import PTBTokenizer

from gezi import Segmentor
segmentor = Segmentor()

import normalize 

tokenizer = None
def tokenization(refs):
  #actually only used for en corpus like MSCOCO, for cn corpus can ignore this
  #for compact also do it for cn corpus 
  print('tokenization...', file=sys.stderr) 
  global tokenizer 
  if tokenizer is None:
    tokenizer = PTBTokenizer() 
    return tokenizer.tokenize(refs)

refs = {}
num = 0
for line in sys.stdin:
  if num % 10000 == 0:
    print(num, file=sys.stderr)
  l = line.rstrip().split('\t')
  img = l[0]
  try:
    texts = l[1].split('\x01')
  except Exception:
    print(line, file=sys.stderr)
    continue
  
  for text in texts:
    text = normalize.norm(text)
    words = segmentor.Segment(text, 'basic')
    if num % 10000 == 0:
      print(text, '|'.join(words), len(words), file=sys.stderr)
    refs.setdefault(img, [])
    refs[img].append(' '.join(words))
  num += 1

#refs = tokenization(refs)

pickle.dump(refs, open(sys.argv[1], 'wb'))

import numpy as np
import dill
from collections import defaultdict
#from cider.py gezi/metrics/cider/cider.py
def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            #print(' '.join(ngram))
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

crefs = []
for _, ref in refs.items():
  crefs.append(cook_refs(ref))

ref_len = np.log(float(len(crefs)))
print('ref_len', len(crefs), ref_len, file=sys.stderr)

document_frequency=defaultdict(float)
for refs in crefs:
  # refs, k ref captions of one image
  for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
    #print(' '.join(ngram))
    document_frequency[ngram] += 1

with open(sys.argv[2], 'w') as f:
  f.write('%f'%ref_len)

print('document_frequency[红毯]', document_frequency[tuple(['红毯'])], file=sys.stderr)
print('document_frequency[(帽子)]', document_frequency[('帽子',)], file=sys.stderr)
print('document_frequency[(戴着 帽子)]', document_frequency[('戴着', '帽子')], file=sys.stderr)
print('document_frequency[(手套)]', document_frequency[('手套',)], file=sys.stderr)
print('document_frequency[(戴着 手套)]', document_frequency[('戴着', '手套')], file=sys.stderr)
dill.dump(document_frequency, open(sys.argv[3], 'wb'))


