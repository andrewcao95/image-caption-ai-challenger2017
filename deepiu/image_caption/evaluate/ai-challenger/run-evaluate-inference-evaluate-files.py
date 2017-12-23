#!/usr/bin/env python
# ==============================================================================
#          \file   run-evaluate-inference-evaluate-dir.py
#        \author   chenghuige  
#          \date   2017-10-07 11:57:11.430650
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import glob 

infile = sys.argv[1]
#for file in glob.glob(dir + '/model*ckpt*.index'):
for file in open(infile):
  file = file.strip().split()[-1]
  file = file.replace('.index', '')
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate-inference-evaluate.py %s'%file 
  print(command, file=sys.stderr)
  os.system(command)
