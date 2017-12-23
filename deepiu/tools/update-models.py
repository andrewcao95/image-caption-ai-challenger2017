#!/usr/bin/env python
# ==============================================================================
#          \file   update-models.py
#        \author   chenghuige  
#          \date   2017-10-08 17:52:47.800893
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob 

dir = sys.argv[1]

for model_file in glob.glob(dir + '/model.ckpt-*.index'):
  model_file = model_file.replace('.index', '')
  model_step = model_file.split('-')[-1]
  new_model_step = str(int(model_step) + 1)
  new_model_file = model_file.replace(model_step, new_model_step)
  command = 'sh /home/gezi/mine/hasky/deepiu/tools/update-model.sh %s 10 0.25'%model_file
  print(command)
  os.system(command)
  final_model_file = new_model_file.replace('model.ckpt', 'model.new.ckpt')
  command = 'mv %s %s'%(new_model_file, final_model_file)
  print(command)
  os.system(command)

