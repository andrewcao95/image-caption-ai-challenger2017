#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   delete-all-epochs-info.py
#        \author   chenghuige  
#          \date   2017-11-07 13:50:02.010709
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

base = '../../../mount'
deal_dir = '%s/temp/image-caption/ai-challenger/model.v4' % base 

dirs = glob.glob(deal_dir + '/*')

print(dirs)

for dir_ in dirs:
  dir_ += '/epoch'
  command = 'rm -rf {0}/events* {0}/log* {0}/summary* {0}/*.evaluate-inference.txt {0}/*.caption*txt {0}/done.txt {0}*.lock'.format(dir_)
  print(command)
  os.system(command)

