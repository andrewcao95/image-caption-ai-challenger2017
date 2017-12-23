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

import sys
import os

import melt
import gezi

assert melt.get_num_gpus() == 1

try:
  model_dir = sys.argv[1] 
except Exception:
  model_dir = './'
model_dir, model_path = melt.get_model_dir_and_path(model_dir)

result_file = model_path + '.evaluate-inference.txt'

print('model_dir', model_dir, 'result_file', result_file, file=sys.stderr)

arg2 = ''
if len(sys.argv) > 2:
  arg2 = sys.argv[2]

if not gezi.non_empty(result_file) or len(open(result_file).readlines()) != 30000:
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/inference/ai-challenger/evaluate-inference.py %s %s' % (model_path, arg2)
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % result_file)

metrics_file = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')

if not gezi.non_empty(metrics_file) or len(open(metrics_file).readlines()) != 30002:
  command = 'CUDA_VISIBLE_DEVICES=-1 python /home/gezi/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate.py %s' % result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % metrics_file)


command = 'CUDA_VISIBLE_DEVICES=-1 /home/gezi/mine/hasky/deepiu/tools/fix-evaluate-summary.py %s' % model_dir
print(command)
os.system(command)
