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
    cdir = os.getcwd()
    dirs = glob.glob('%s/*/epoch' % cdir)
    #random.shuffle(dirs)
  else:
    dirs = sys.argv[1].split(',') 
  dirs.sort(key=os.path.getmtime, reverse=prefer_new)
    
  print('dirs', dirs)
  for dir in dirs:
    local_dir = dir.replace('hdfsmount', 'mount')
    command = 'mkdir -p %s' % local_dir 
    print(command, file=sys.stderr)
    os.system(command)
    command = 'python /home/gezi/mine/hasky/deepiu/tools/remote-monitor-dir.py %s --do_once=1 --prefer_new %d --start_epoch %d' % (dir, prefer_new, start_epoch)
    print(command, file=sys.stderr)
    os.system(command)
  time.sleep(10)
    
