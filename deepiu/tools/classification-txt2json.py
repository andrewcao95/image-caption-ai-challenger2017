#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calssification-txt2json.py
#        \author   chenghuige  
#          \date   2017-11-10 16:42:47.665378
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

#import json
#import codesc

reload(sys)
sys.setdefaultencoding('utf8')
import json
import io

res = []
infile = sys.argv[1]
for line in open(infile):
  l = line.strip().split('\t')
  pic, classes = l[0], l[1]
  classes = [int(x) for x in classes.split()]
  res.append({'image_id': pic, 'label_id': classes[:3]})

if len(sys.argv) > 2:
  outfile = sys.argv[2]
else:
  outfile = infile.replace('.txt', '.json')
#json.dump(res, codesc.open(sys.stdout, 'w', 'utf8'), ensure_ascii=False, indent=4)
with io.open(outfile, 'w', encoding='utf-8') as fd:
  fd.write(unicode(json.dumps(res, ensure_ascii=False, sort_keys=True, 
                              indent=2, separators=(',', ': '))))