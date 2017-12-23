#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate-inference-evaluate.py
#        \author   chenghuige  
#          \date   2017-10-06 21:09:24.398921
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import melt

model_dir = sys.argv[1] 
model_path = melt.get_model_path(model_dir)

result_file = model_path + '.evaluate-inference.txt'

text_key = sys.argv[2]
score_key = sys.argv[3]

if not os.path.exists(result_file):
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/inference/ai-challenger/evaluate-inference2.py %s %s %s'%(model_path, text_key, score_key)
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists'%result_file)

result_file2 = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')

if not os.path.exists(result_file2):
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate.py %s'%result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists'%result_file2)
