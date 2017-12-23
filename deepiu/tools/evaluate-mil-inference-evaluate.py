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

input_file = sys.argv[2]

result_file = model_path + '.evaluate-inference.txt'

batch_size = 50
if len(sys.argv) > 3:
  batch_size = int(sys.argv[3])

feature_name = 'attention'
if len(sys.argv) > 4:
  feature_name = sys.argv[4]

if not gezi.non_empty(result_file) or len(open(result_file).readlines()) != 30000:
  if feature_name:
    command = 'python /home/gezi/mine/hasky/deepiu/tools/evaluate-mil-inference.py %s %s %d %s' % (model_path, input_file, batch_size, feature_name)
  else:
    command = 'python /home/gezi/mine/hasky/deepiu/tools/evaluate-mil-inference.py %s %s %d' % (model_path, input_file, batch_size)
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % result_file)

metrics_file = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')

if not gezi.non_empty(metrics_file):
  command = 'CUDA_VISIBLE_DEVICES=-1 python /home/gezi/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate.py %s' % result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % metrics_file)

