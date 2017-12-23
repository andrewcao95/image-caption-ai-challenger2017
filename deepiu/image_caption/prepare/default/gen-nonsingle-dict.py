#!/usr/bin/env python
# ==============================================================================
#          \file   gen-nonsingle-dict.py
#        \author   chenghuige  
#          \date   2017-09-25 16:59:29.786043
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import gezi

f1 = sys.argv[1]
f2 = sys.argv[2]

out = open(sys.argv[3], 'w')

m = {}

def deal_file(f):
  for line in open(f):
    word, count  = line.rstrip().split('\t')
    count = int(count)
  
    cns = gezi.get_single_cns(word)
  
    if len(cns) <= 1:
      continue
  
    m.setdefault(word, 0)
    m[word] += count 


deal_file(f1)

deal_file(f2)


l = [(count, word) for word, count in m.items()]
l.sort(reverse=True)
for count, word in l:
  print(word, count, sep='\t', file=out)
