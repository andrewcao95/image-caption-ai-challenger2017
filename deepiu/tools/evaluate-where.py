#!/usr/bin/env python
# ==============================================================================
#          \file   monitor_epoch.py
#        \author   chenghuige  
#          \date   2017-09-19 12:14:39.924016
#   \Description  
# ==============================================================================

import sys, os

command = 'cat %s > /tmp/temp.txt'%sys.argv[1]
print command
os.system(command)

command = """
python /home/gezi/tools/where.py /tmp/temp.txt \'%s\' /tmp/result.txt
  """%sys.argv[2]
print command
os.system(command)

command = 'python /home/gezi/mine/hasky/deepiu/tools/evaluate-caption2html.py /tmp/result.txt case.html'
print command
os.system(command)
