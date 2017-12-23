#!/usr/bin/env python
# ==============================================================================
#          \file   monitor-dirs.py
#        \author   chenghuige  
#          \date   2017-10-19 14:19:47.395408
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os	
import time
import random
import glob
import melt 

num_gpus = melt.get_num_gpus()
assert num_gpus == 1

prefer_new = True

if len(sys.argv) > 2:
  #need int!
  prefer_new = bool(int(sys.argv[2])) 

start_epoch=0
if len(sys.argv) > 3:
  start_epoch = int(sys.argv[3])

while True:
  if sys.argv[1] == '.':
    dirs = glob.glob('./*/epoch')
    random.shuffle(dirs)
    dirs.sort(key=os.path.getmtime, reverse=prefer_new)
  else:
    dirs = sys.argv[1].split(',')
  for dir in dirs:
    command = 'python /home/gezi/mine/hasky/deepiu/tools/monitor-dir.py %s --do_once=1 --prefer_new %d --start_epoch %d' % (dir, prefer_new, start_epoch)
    print(command, file=sys.stderr)
    os.system(command)
  time.sleep(10)
    
