#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================
"""
@TODO using logging ?
NOTICE using feed_dict --feed_neg=1 will slow *10
@TODO why evaluate will use mem up to 13% then down to 5%?
using tf.nn.top_k do sort in graph still evaluate large mem same as before
20w will using 1.6G 32 * 5%
50w 9.1%
200w 16G?
2000w 160g?

@TODO why using gpu will use more cpu mem?
50w keyword MAX_TEXT_WORDS 20
cpu version: 
show_eval=0  train 6.7% eval 11.4%
show_eval=1 train 8% eval 20%

gpu version: 
show_eval=0 train 8% eval 12.8%
show_eval=1 train 35% eval 39%

text means text ids
text_str means orit text str
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, '''actually model_path is better,
                                        The directory where the model was written to or 
                                        an absolute path to a checkpoint file''')
flags.DEFINE_string('algo', 'bow', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')
flags.DEFINE_boolean('debug', False, '')

import sys, os
import functools
import gezi
import melt
logging = melt.logging

from deepiu.image_caption import input_app as InputApp
from deepiu.image_caption import eval_show

from deepiu.util import evaluator
from deepiu.util import algos_factory
from deepiu.util import image_util

#debug
from deepiu.util import vocabulary
from deepiu.util import text2ids
from deepiu.util import reinforcement_learning

from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

import tensorflow.contrib.slim as slim

from deepiu.image_caption.conf import IMAGE_FEATURE_LEN

import traceback, copy

import numpy as np

import inspect

sess = None

#feed ops maninly for inital state re feed, see ptb_word_lm.py, notice will be much slower adding this
# state = m.initial_state.eval()
# for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
#                                                   m.num_steps)):
#   cost, state, _ = session.run([m.cost, m.final_state, eval_op],
#                                {m.input_data: x,
#                                 m.targets: y,
#                                 m.initial_state: state})
feed_ops = []
feed_run_ops = []
feed_results = []

gtrain_image_name = None
geval_image_name = None

names = ['loss', 'main_loss', 'scene_loss', 'scene_recall@1', 'scene_recall@3']
eval_names = None

#TODO do not consider feed dict support right now for tower loss!
def tower_loss(trainer, input_app=None, input_results=None):
  if input_app is None:
    input_app = InputApp.InputApp()
  if input_results is None:
    input_results = input_app.gen_input(train_only=True)

  #--------train
  weights = None
  if not FLAGS.use_weights:
    image_name, image_feature, text, text_str = input_results[input_app.input_train_name]
  else:
    image_name, image_feature, text, text_str, weights = input_results[input_app.input_train_name]

  global gtrain_image_name #for rl
  gtrain_image_name = image_name

  with tf.device('/gpu:0'):
    main_loss = trainer.build_train_graph(image_feature, text)

  weights = None
  if not FLAGS.use_weights:
    image_name, image, label, text_str = input_results[input_app.scene_input_train_name]
  else:
    image_name, image, label, text_str, weights = input_results[input_app.scene_input_train_name]
  
  with tf.device('/gpu:1'):
    scene_loss = trainer.build_scene_graph(image, label)

  loss = main_loss + 0.1 * scene_loss
  return loss, main_loss, scene_loss

def gen_train_graph(input_app, input_results, trainer):
  """
    main flow, key graph
  """
  #--- if you don't want to use mutli gpu, here just for safe(code same with old single gpu cod)
  
  loss, main_loss, scene_loss = tower_loss(trainer, input_app, input_results)

  ops = [loss, main_loss, scene_loss, trainer.scene_recall_at_1, trainer.scene_recall_at_k]
    
  deal_debug_results = None
  if FLAGS.debug == True:
    ops += [tf.get_collection('train_scores')[-1]]
    if FLAGS.discriminant_loss_ratio > 0:
      ops += [tf.get_collection('gen_loss')[-1], tf.get_collection('discriminant_loss')[-1]]
      #ops += [tf.get_collection('pos_score')[-1], tf.get_collection('neg_score')[-1]]   
      ops += [tf.get_collection('pos_score')[-1]]
    if FLAGS.alignment_history:
      ops += [tf.get_collection('alignment_history')[-1]]
      if FLAGS.alignment_loss_ratio > 0:
        ops += [tf.get_collection('alignment_loss')[-1]]
    if FLAGS.use_weights or FLAGS.use_idf_weights:
      ops += [tf.get_collection('targets')[-1]]
      ops += [tf.get_collection('mask')[-1]]
    def _deal_debug_results(results):
      if FLAGS.use_weights or FLAGS.use_idf_weights:
        # NOTICE need FLAGS.train_loss_per_example = True, so will not flatten losses
        print('targets:', text2ids.ids2text(results[-2][0]))
        print('mask:', results[-1][0])
      print('results:', [x for x in results if len(x.shape) == 0])
      print('shapes:', [x.shape for x in results])

    deal_debug_results = _deal_debug_results


  return ops, deal_debug_results

counter = 0
def gen_train(input_app, input_results, trainer):  
  ops, deal_debug_results = gen_train_graph(input_app, input_results, trainer)
  
  #@NOTICE make sure global feed dict ops put at last
  if hasattr(trainer, 'feed_ops'):
    global feed_ops, feed_run_ops, feed_results
    feed_ops, feed_run_ops = trainer.feed_ops()
    #feed_results = sess.run(feed_run_ops)
    ops += feed_run_ops

  def _deal_results(results):
    if FLAGS.debug == True:
      melt.print_results(results, ['loss'])

    if deal_debug_results is not None:
      debug_results = results[:-len(feed_ops)] if feed_ops else results
      deal_debug_results(debug_results)

    if feed_ops:
      global feed_results
      feed_results = results[-len(feed_ops):]

  deal_results = _deal_results


  def _gen_feed_dict():
    feed_dict = {}
    
    if feed_results:
      for op, result in zip(feed_ops, feed_results):
        feed_dict[op] = result
        #print(op, result)
    return feed_dict

  def _gen_rl_feed_dict():
    global counter 
    counter += 1 

    image_names, sampled_captions, greedy_captions = sess.run([gtrain_image_name, trainer.rl.sampled_caption, trainer.rl.greedy_caption])
    #notice suggest not using tokenize, especially for cn , no use and will print log..
    rewards, baseline = reinforcement_learning.calc_score(sampled_captions, greedy_captions, image_names, tokenize=FLAGS.use_tokenize)
    if counter % 100 == 0:
      logging.info('label__caption: {}'.format('|'.join(evaluator.refs[image_names[0]])))
      logging.info('sample_caption: {}'.format(text2ids.ids2text(sampled_captions[0])))
      logging.info('greedy_caption: {}'.format(text2ids.ids2text(greedy_captions[0])))
      logging.info('rewards: {} baseline: {}'.format(rewards[0], baseline[0]))
    feed_dict = {trainer.rl.rewards_feed: rewards, trainer.rl.baseline_feed: baseline}
    return feed_dict

  def _gen_all_feed_dict():
    feed_dict = _gen_feed_dict()
    if FLAGS.reinforcement_learning:
      feed_dict = gezi.merge_dicts(feed_dict, _gen_rl_feed_dict())
    return feed_dict

  # if not FLAGS.reinforcement_learning:
  #   gen_feed_dict = _gen_feed_dict  
  # else:
  #   gen_feed_dict = _gen_rl_feed_dict

  gen_feed_dict = _gen_all_feed_dict

  return ops, gen_feed_dict, deal_results

def gen_evalulate(input_app, 
                  input_results,
                  predictor, 
                  eval_ops, 
                  eval_scores,
                  eval_neg_text=None,
                  eval_neg_text_str=None):
  if algos_factory.is_discriminant(FLAGS.algo):
    eval_ops += eval_show.gen_eval_show_ops(
        input_app, 
        input_results, 
        predictor, 
        eval_scores, 
        eval_neg_text, 
        eval_neg_text_str)
    deal_eval_results = eval_show.deal_eval_results
  else:
    eval_ops += eval_show.gen_eval_generated_texts_ops(
        input_app, 
        input_results, 
        predictor, 
        eval_scores,
        eval_neg_text,
        eval_neg_text_str)
    deal_eval_results = eval_show.deal_eval_generated_texts_results
  return eval_ops, deal_eval_results

def gen_validate(input_app, input_results, validator, predictor):
  gen_eval_feed_dict = None
  eval_ops = None
  train_with_validation = input_results[input_app.input_valid_name] is not None
  scene_train_with_validation = input_results[input_app.scene_input_valid_name] is not None
  deal_eval_results = None
  if train_with_validation and not FLAGS.train_only:
    eval_image_name, eval_image_feature, eval_text, eval_text_str = input_results[input_app.input_valid_name]
    global geval_image_name
    geval_image_name = eval_image_name

    if input_results[input_app.input_valid_neg_name]:
      eval_neg_image_name, eval_neg_image_feature, eval_neg_text, eval_neg_text_str = input_results[input_app.input_valid_neg_name]
      
    if not FLAGS.neg_left:
      eval_neg_image_feature = None 
    eval_neg_text_ = eval_neg_text
    if not FLAGS.neg_right:
      eval_neg_text_ = None
    if algos_factory.is_generative(FLAGS.algo):
      eval_neg_image_feature = None
      eval_neg_text_ = None
    
    global eval_names
    with tf.device('/gpu:0'):
      eval_main_loss = validator.build_train_graph(eval_image_feature, eval_text, eval_neg_image_feature, eval_neg_text_)
    eval_scores = tf.get_collection('eval_scores')[-1]
    eval_ops = [eval_main_loss]
    eval_names = ['eval_main_loss']

    if scene_train_with_validation:
      eval_image_name, eval_image, eval_label, eval_text_str = input_results[input_app.scene_input_valid_name]

      with tf.device('/gpu:1'):
        eval_scene_loss = validator.build_scene_graph(eval_image, eval_label)
      eval_loss = eval_main_loss + eval_scene_loss
      eval_ops.insert(0, eval_loss)
      eval_names.insert(0, 'eal_loss')
      eval_ops += [eval_scene_loss, validator.scene_recall_at_1, validator.scene_recall_at_k]
      eval_names += ['eval_scene_loss', 'eval_scene_recall@1', 'eval_scene_recall@3']

    if algos_factory.is_generative(FLAGS.algo):
      eval_neg_text = None 
      eval_neg_text_str = None 
      
    if FLAGS.show_eval and (predictor is not None):
      eval_ops, deal_eval_results = \
        gen_evalulate(
            input_app, 
            input_results, 
            predictor, 
            eval_ops, 
            eval_scores,
            eval_neg_text, 
            eval_neg_text_str)
    else:
      #deal_eval_results = None
      def _deal_eval_results(results):
        print(results)
      deal_eval_results = _deal_eval_results

  def _gen_feed_dict():
    feed_dict = {}
    return feed_dict

  def _gen_scene_feed_dict():
    return gen_scene_feed_dict(validator, geval_image_name)

  def _gen_all_feed_dict():
    feed_dict = _gen_feed_dict()
    if FLAGS.scene_model:
      feed_dict = gezi.merge_dicts(feed_dict, _gen_scene_feed_dict())
    return feed_dict

  gen_eval_feed_dict = _gen_all_feed_dict
  return eval_ops, gen_eval_feed_dict, deal_eval_results

def gen_predict_graph(predictor):  
  """
  call it at last , build predict graph
  the probelm here is you can not change like beam size later...
  """
  #-----discriminant and generative
  predictor.init_predict() #here self add all score ops
  #-----generateive
  if algos_factory.is_generative(FLAGS.algo):
    exact_score = predictor.init_predict(exact_loss=True)
    tf.add_to_collection('exact_score', exact_score)
    exact_prob = predictor.init_predict(exact_prob=True)
    tf.add_to_collection('exact_prob', exact_prob)

    ##TODO
    # beam_size = tf.placeholder_with_default(FLAGS.beam_size, shape=None)
    # tf.add_to_collection('beam_size_feed', beam_size)
    
    init_predict_text = predictor.init_predict_text

    text, text_score = init_predict_text(decode_method=FLAGS.seq_decode_method) #greedy

    tf.add_to_collection('text', text)
    tf.add_to_collection('text_score', text_score)

    try:
      init_predict_text(decode_method=SeqDecodeMethod.outgraph_beam,
                        beam_size=FLAGS.beam_size) #outgraph
    except Exception:
      print(traceback.format_exc(), file=sys.stderr)
      print('warning: outgraph beam search not supported', file=sys.stderr)
      pass
    
    full_text, full_text_score = init_predict_text(decode_method=SeqDecodeMethod.ingraph_beam,
                                                   beam_size=100)    
    tf.add_to_collection('full_text', full_text)
    tf.add_to_collection('full_text_score', full_text_score)  

    #if not FLAGS.reinforcement_learning:
    beam_text, beam_text_score = init_predict_text(decode_method=SeqDecodeMethod.ingraph_beam,
                                                   beam_size=FLAGS.beam_size,
                                                   logprobs_history=True, 
                                                   alignment_history=True) #ingraph must at last 
    #else:
    #  beam_text, beam_text_score = tf.expand_dims(text, 1), tf.expand_dims(text_score, 1)

    tf.add_to_collection('beam_text', beam_text)
    tf.add_to_collection('beam_text_score', beam_text_score)
    tf.add_to_collection('beam_logprobs_history', predictor.decoder.log_probs_history)
    tf.add_to_collection('beam_alignment_history', predictor.decoder.alignment_history)
    predictor.logprobs_history = False 
    predictor.alignment_history = False

def train():
  input_app = InputApp.InputApp()
  input_results = input_app.gen_input()
  global_scope = melt.apps.train.get_global_scope()

  with tf.variable_scope(global_scope) as global_scope:
    with tf.variable_scope(FLAGS.main_scope) as scope:
      trainer, validator, predictor = algos_factory.gen_all(FLAGS.algo)
      
      if FLAGS.reinforcement_learning:
        from deepiu.image_caption.algos.show_and_tell import RLInfo
        trainer.rl = RLInfo()

      if FLAGS.scene_model:
        #TODO scene model not support show_eval right now due to batch size eg 3 for show_eval
        from deepiu.image_caption.algos.show_and_tell import SceneInfo 
        trainer.scene = SceneInfo(FLAGS.batch_size)
        validator.scene = SceneInfo(FLAGS.eval_batch_size)
        predictor.scene = SceneInfo()

      logging.info('trainer:{}'.format(trainer))
      logging.info('predictor:{}'.format(predictor))

      ops, gen_feed_dict, deal_results = gen_train(
      input_app, 
      input_results, 
      trainer)
    
      scope.reuse_variables()

      #saving predict graph, so later can direclty predict without building from scratch
      #also used in gen validate if you want to use direclty predict as evaluate per epoch
      if predictor is not None and FLAGS.gen_predict:
        gen_predict_graph(predictor)

      eval_ops, gen_eval_feed_dict, deal_eval_results = gen_validate(
        input_app, 
        input_results, 
        validator, 
        predictor)

      metric_eval_fn = None
      if FLAGS.metric_eval:
        if FLAGS.scene_model:
          evaluator.gen_feed_dict_fn = lambda image_features: gen_predict_scene_feed_dict(predictor, image_features)

        eval_rank = FLAGS.eval_rank and (not algos_factory.is_generative(FLAGS.algo) or FLAGS.assistant_model_dir) 
        eval_translation = FLAGS.eval_translation and algos_factory.is_generative(FLAGS.algo)
        metric_eval_fn = lambda: evaluator.evaluate(predictor, random=True, eval_rank=eval_rank, eval_translation=eval_translation)

  # NOTCIE in empty scope now, image model need to escape all scopes!
  summary_excls = None
  init_fn, restore_fn = image_util.get_init_restore_fn()

  with tf.variable_scope(global_scope):
    #melt.print_global_varaiables()
    melt.apps.train_flow(ops, 
                        names=names,
                        gen_feed_dict_fn=gen_feed_dict,
                        deal_results_fn=deal_results,
                        eval_ops=eval_ops,
                        eval_names=eval_names,
                        gen_eval_feed_dict_fn=gen_eval_feed_dict,
                        deal_eval_results_fn=deal_eval_results,
                        optimizer=FLAGS.optimizer,
                        learning_rate=FLAGS.learning_rate,
                        num_steps_per_epoch=input_app.num_steps_per_epoch,
                        metric_eval_fn=metric_eval_fn,
                        summary_excls=summary_excls,
                        init_fn=init_fn,
                        restore_fn=restore_fn,
                        sess=sess)  # notice if use melt.constant in predictor then must pass sess

def main(_):
  #-----------init global resource
  melt.apps.train.init()

  FLAGS.vocab = FLAGS.vocab or os.path.join(os.path.dirname(FLAGS.model_dir), 'vocab.txt')
  
  image_util.init()

  vocabulary.init()
  text2ids.init()

  ## TODO FIXME if evaluator init before main graph(assistant predictor with image model) then will wrong for finetune later,
  ## image scope as not defined, to set reuse = None? though assistant in different scope graph still scope reused??
  ## so right now just let evaluator lazy init, init when used after main graph build

  # try:
  #   evaluator.init()
  # except Exception:
  #   print(traceback.format_exc(), file=sys.stderr)
  #   print('evaluator init fail will not do metric eval')
  #   FLAGS.metric_eval = False

  logging.info('algo:{}'.format(FLAGS.algo))
  logging.info('monitor_level:{}'.format(FLAGS.monitor_level))

  global sess
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)

  train()
 
if __name__ == '__main__':
  tf.app.run()
