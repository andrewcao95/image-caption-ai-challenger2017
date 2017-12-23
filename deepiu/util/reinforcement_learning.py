#!/usr/bin/env python
# ==============================================================================
#          \file   reinforcement_learning.py
#        \author   chenghuige  
#          \date   2017-09-23 22:43:24.268845
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags 
#import gflags as flags
FLAGS = flags.FLAGS

flags.DEFINE_string('rl_metrics', 'cider', 'or bleu_4 or meteor or rouge_l or cider,bleu_4 will calc mean score')

import sys, os
import numpy as np
from deepiu.util import evaluator

START_KEY = '<S>'
END_KEY = '</S>'

def calc_score(lcaption, rcaption, image_name, tokenize=False):
	evaluator.init()
	refs = evaluator.prepare_refs()

	lresults = evaluator.translation_ids2words(lcaption, image_name)
	rresults = evaluator.translation_ids2words(rcaption, image_name)

	selected_refs = {}
	selected_lresults = {}
	selected_rresults = {}
	#by doing this can force same .keys()

	for i in range(len(lresults)):
		#selected_refs[i] = refs[image_name[i]]
		#need end mark so to punisment generating long sentences with out end mark
		#selected_refs[i] = ['%s %s %s'%(START_KEY, x, END_KEY) for x in refs[image_name[i]]]
		selected_refs[i] = [x for x in refs[image_name[i]]]

		selected_lresults[i] = lresults[i]
		selected_rresults[i] = rresults[i]

	if tokenize:
	 	selected_lresults = evaluator.tokenization(selected_lresults)
	 	selected_rresults = evaluator.tokenization(selected_rresults)

	#print('selected_lresults len:', len(selected_lresults))
	#print('lresult:', selected_lresults[0][0], 'rresult:', selected_rresults[0][0])
	#print('ref:', '|'.join(selected_refs[0]))
	lscores = evaluator.translation_scores(selected_lresults, selected_refs, FLAGS.rl_metrics)
	rscores = evaluator.translation_scores(selected_rresults, selected_refs, FLAGS.rl_metrics)


	#print(selected_refs.keys())
	lscores = np.array(lscores)
	rscores = np.array(rscores)
	
	#print('-----', lscores)
	#print('----', lscores - rscores)
	#print(lscores, rscores)
	return lscores, rscores

  
