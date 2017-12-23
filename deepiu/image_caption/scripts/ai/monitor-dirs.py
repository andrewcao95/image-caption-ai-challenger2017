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
  prefer_new = bool(int(sys.argv[1])) 

base = '../../../mount/temp/image-caption/ai-challenger/'
def get_monitor_path():
  monitor_path = '%s/model.v4' % base 
  if os.path.exists('monitor.txt'):
    monitor_path = '%s/%s' % (base, open('monitor.txt').readline().strip())
  return monitor_path

def get_dirs():
  monitor_dir = get_monitor_path()
  dirs = glob.glob('%s/*/epoch' % monitor_dir)
  random.shuffle(dirs)
  return dirs 

while True:
  dirs = get_dirs()
  dirs.sort(key=os.path.getmtime, reverse=prefer_new)
  print(dirs, len(dirs))
  for dir in dirs:
    #if not 'inceptionV4' in dir or not 'finetune' in dir:
    #  continue
    command = 'python ./scripts/ai/monitor-dir.py %s --do_once=1 --prefer_new %d' % (dir, prefer_new)
    print(command, file=sys.stderr)
    os.system(command)
  time.sleep(10)
    
