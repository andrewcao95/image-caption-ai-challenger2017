#!/usr/bin/env python
# ==============================================================================
#          \file   catpion-txt2json.py
#        \author   chenghuige  
#          \date   2017-09-28 07:31:34.052898
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
  pic, caption = l[0], l[1]
  res.append({'caption': caption, 'image_id': pic})

if len(sys.argv) > 2:
  outfile = sys.argv[2]
else:
  outfile = infile.replace('.txt', '.json')
#json.dump(res, codesc.open(sys.stdout, 'w', 'utf8'), ensure_ascii=False, indent=4)
with io.open(outfile, 'w', encoding='utf-8') as fd:
  fd.write(unicode(json.dumps(res, ensure_ascii=False, sort_keys=True, 
                              indent=2, separators=(',', ': '))))
