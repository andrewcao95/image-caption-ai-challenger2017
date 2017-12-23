#!/usr/bin/env python
# ==============================================================================
#          \file   ensemble-inference.py
#        \author   chenghuige  
#          \date   2017-10-06 21:29:49.558024
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('start_epoch', 0, '')
flags.DEFINE_integer('epoch_interval', 1, '')
flags.DEFINE_boolean('do_once', False, '')
flags.DEFINE_boolean('prefer_new', True, '')

import sys, os, time, glob

import melt

try:
  dir = sys.argv[1]
except Exception:
  dir = './'
  
done_files_name = '%s/done.txt' % dir

if not os.path.exists(done_files_name):
  command = 'touch %s' % done_files_name
  print(command, file=sys.stderr)
  os.system(command)

num_gpus = melt.get_num_gpus()
assert num_gpus == 1

FLAGS.prefer_new = True
print('prefer_new:', FLAGS.prefer_new)

while True:
  suffix = '.data-00000-of-00001'
  files = glob.glob(os.path.join(dir, 'model.ckpt*.data-00000-of-00001'))
  #print(files)
  if not FLAGS.prefer_new:
    files.sort(key=os.path.getmtime)
  else:
    files.sort(key=os.path.getmtime, reverse=True)
  #from epoch 1, 2, ..
  #files.sort(key=os.path.getmtime)
  files = [file.replace(suffix, '') for file in files if not 'best' in file.split('/')[-1]]
  #print(files)

  #break

  for file in files:
    file_ = os.path.basename(file)
    #print(file)
    #break
    if not os.path.exists(file + '.meta') or not os.path.exists(file + '.index'):
      continue

    epoch = int(float(file_.split('-')[1]))
    #step = int(float(file.split('-')[2]))
    if FLAGS.start_epoch and epoch < FLAGS.start_epoch:
      continue
    if FLAGS.epoch_interval and epoch % FLAGS.epoch_interval != 0:
      continue

    lock_file = '%s.lock' % file 
    if os.path.exists(lock_file):
      print('%s exists' % lock_file)
      continue
    command = 'touch %s.lock' % file
    print(command)
    os.system(command)

    command = 'python ./scripts/ai/evaluate-inference-evaluate.py %s' % file
    print(command, file=sys.stderr)
    os.system(command)
    
    if os.path.exists(lock_file):
      print('done and remove %s' % lock_file)
      os.remove(lock_file)

    time.sleep(5)
    if FLAGS.do_once:
      break

  if FLAGS.do_once:
    break

  #break
  time.sleep(10)
