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

print('prefer_new:', FLAGS.prefer_new)

while True:
  done_files = [x.strip() for x in open(done_files_name).readlines()]
  #print('done_files', done_files)
  suffix = '.index'
  files = glob.glob(os.path.join(dir, 'model.ckpt*.index'))
  #print(files)
  files.sort(key=os.path.getmtime, reverse=FLAGS.prefer_new)
  #from epoch 1, 2, ..
  #files.sort(key=os.path.getmtime)
  files = [file.replace(suffix, '') for file in files if not 'best' in file.split('/')[-1]]
  #print(files)

  #break

  for file in files:
    file_ = os.path.basename(file)
    #if file_ in done_files and os.path.exists(file + '.evaluate-inference.txt') and len(open(file + '.evaluate-inference.txt').readlines()) == 30000:
    if os.path.exists(file + '.evaluate-inference.txt') and len(open(file + '.evaluate-inference.txt').readlines()) == 30000 and \
            os.path.exists(file + '.caption_metrics.txt') and len(open(file + '.caption_metrics.txt').readlines()) == 30002:
      #print('exists', file)
      #if not FLAGS.prefer_new:
      continue
      #else:
        #if latest is done do not process older ones
      #  break
    #print(file)
    #break 
    ok = False 
    if os.path.exists(file + '.pb') or os.path.exists(file + '.meta'):
      ok = True 

    if not ok:
      continue

    epoch = int(float(file_.split('-')[1]))
    #step = int(float(file.split('-')[2]))
    # HACK  TODO
    if FLAGS.start_epoch and epoch < FLAGS.start_epoch:
      if not FLAGS.prefer_new:
        continue
      else:
        break
    if FLAGS.epoch_interval and epoch % FLAGS.epoch_interval != 0:
      continue

    lock_file = '%s.lock' % file 
    if os.path.exists(lock_file):
      print('%s exists' % lock_file)
      continue
      #break
    command = 'touch %s.lock' % file
    print(command)
    os.system(command)

    command = '/home/gezi/mine/hasky/deepiu/tools/evaluate-inference-evaluate.py %s' % file
    print(command, file=sys.stderr)
    os.system(command)
    
    if len(open(file + '.evaluate-inference.txt').readlines()) == 30000:
      with open(done_files_name, 'a') as out:
        print(file_, file=out)
    
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
