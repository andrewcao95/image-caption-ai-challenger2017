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
import melt, gezi

assert melt.get_num_gpus() == 1

try:
  model_dir = sys.argv[1] 
except Exception:
  model_dir = './'
model_path = melt.get_model_path(model_dir)

result_file = model_path + '.evaluate-inference.txt'

if not gezi.non_empty(result_file):
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/inference/ai-challenger/evaluate-inference.py %s' % model_path 
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % result_file)

metrics_file = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')

if not gezi.non_empty(metrics_file):
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate.py %s' % result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % metrics_file)

inference_file = result_file.replace('.evaluate-inference.txt', '.inference.txt')
if not gezi.non_empty(inference_file):
  command = 'python /home/gezi/mine/hasky/deepiu/tools/run-inference.py %s' % model_path 
  print(command)
  os.system(command)
