#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate-inference.py
#        \author   chenghuige  
#          \date   2017-10-06 20:29:49.144771
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import gezi
import melt

model_dir = sys.argv[1]

model_dir, model_path = melt.get_model_dir_and_path(model_dir)

result_file = model_path + '.inference.txt'

if gezi.non_empty(result_file):
  print('%s already exists, do nothing' % result_file)
  exit(0)

def has_model(model_dir, model_name):
  return model_name in model_dir or model_name in os.getcwd()

batch_size=300
if has_model(model_dir, 'nasnet'):
  #batch_size = 100
  batch_size = 50  #  for nasnet must use 50 the same as training(eval size).... otherwise wrong result TODO FIXME

num_gpus = melt.get_num_gpus()
assert num_gpus == 1

command = """python /home/gezi/mine/hasky/deepiu/tools/classification-inference.py \\
  --model_dir={} \\
  --test_image_dir=/home/gezi/data2/data/ai_challenger/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/ \\
  --buffer_size={} \\
  --result_file={} \\
  """.format(model_path, batch_size, result_file)

print(command)

os.system(command)

