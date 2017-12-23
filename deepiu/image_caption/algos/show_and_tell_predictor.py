#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell_predictor.py
#        \author   chenghuige  
#          \date   2016-09-04 17:50:21.017234
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

import os 
import functools
import numpy as np

import gezi
import melt

from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS
from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts
from deepiu.image_caption.algos.show_and_tell import ShowAndTell

from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

class ShowAndTellPredictor(ShowAndTell, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    ShowAndTell.__init__(self, is_training=False, is_predict=True)

    if FLAGS.pre_calc_image_feature:
      self.image_feature_len = FLAGS.image_feature_len or IMAGE_FEATURE_LEN 
      #TODO for rl, need use feed dict, so predict will introduce ... need to feed, how to use with_default?
      #self.image_feature_feed = tf.placeholder(tf.float32, [None, self.image_feature_len], name='image_feature')
      self.image_feature_feed = tf.placeholder_with_default([[0.] * self.image_feature_len], [None, self.image_feature_len], name='image_feature')
    else:
      #self.image_feature_feed =  tf.placeholder(tf.string, [None,], name='image_feature')
      # TODO HACK for nasnet... need this due to using average decay
      if os.path.exists('./test.jpg'):
        test_image = melt.read_image('./test.jpg')
      elif os.path.exists('/tmp/test.jpg'):
        test_image = melt.read_image('/tmp/test.jpg')
      else:
        test_image = None
      
      if test_image is not None:
        self.image_feature_feed = tf.placeholder_with_default(tf.constant([test_image]), [None, ], name='image_feature')
      else:
        assert not FLAGS.image_model_name.startswith('nasnet'), 'HACK for nasnet you need one test.jpg in current path or /tmp/ path' 
        self.image_feature_feed = tf.placeholder(tf.string, [None, ], name='image_feature')

    tf.add_to_collection('feed', self.image_feature_feed)
    tf.add_to_collection('lfeed', self.image_feature_feed)
    
    self.text_feed = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')
    tf.add_to_collection('rfeed', self.text_feed)

    self.text = None 
    self.text_score = None
    self.beam_text = None 
    self.beam_text_score = None

    self.image_model = None

    self.logprobs_history = False 
    self.alignment_history = False

    self.feed_dict = {}

  def init_predict_text(self, 
                        decode_method='greedy', 
                        beam_size=5, 
                        convert_unk=False, 
                        length_normalization_factor=None,
                        max_words=None,
                        logprobs_history=False,
                        alignment_history=False):
    """
    init for generate texts
    """
    self.logprobs_history = logprobs_history 
    self.alignment_history = alignment_history 

    text, score = self.build_predict_text_graph(self.image_feature_feed, 
      decode_method, 
      beam_size, 
      convert_unk,
      length_normalization_factor=length_normalization_factor,
      max_words=max_words, 
      logprobs_history=logprobs_history,
      alignment_history=alignment_history)

    if decode_method == SeqDecodeMethod.greedy:
      self.text, self.text_score = text, score 
    elif decode_method == SeqDecodeMethod.ingraph_beam:
      self.beam_text, self.beam_text_score = text, score 
    return text, score

  def init_predict(self, exact_prob=False, exact_loss=False):
    self.score = self.build_predict_graph(self.image_feature_feed, 
                                          self.text_feed, 
                                          exact_prob=exact_prob,
                                          exact_loss=exact_loss)
    tf.add_to_collection('score', self.score)
    
    # self.dupimage = True
    # self.dupimage_score = self.build_predict_graph(self.image_feature_feed, 
    #                                       self.text_feed, 
    #                                       exact_prob=exact_prob,
    #                                       exact_loss=exact_loss)
    # tf.add_to_collection('dupimage_score', self.dupimage_score)
    # self.dupimage = False
    
    return self.score

  #make different name compare to shown_and_tell.encode
  def _encode(self, image):
    attention_states, initial_state, image_emb = self.encoder.encode(self.process(image))
    if FLAGS.image_as_init_state:
      #for im2txt one more step at first
      with tf.variable_scope(self.decoder.scope):
        batch_size=melt.get_batch_size(image_emb)
        zero_state = self.decoder.cell.zero_state(batch_size, dtype=tf.float32)
        _, initial_state = self.decoder.cell(image_emb, zero_state)
        image_emb = self.decoder.get_start_embedding_input(batch_size)
    elif image_emb is None:
      #TODO check
      batch_size=melt.get_batch_size(image)
      image_emb = self.decoder.get_start_embedding_input(batch_size)

    image_emb = self._post_deal_image_embedding(image_emb, image)
    #attention_states, initial_state, image_emb = self._post_deal(attention_states, initial_state, image_emb)
    return attention_states, initial_state, image_emb

  #TODO Notice when training this image always be float...  
  def build_predict_text_graph(self, image, decode_method='greedy', beam_size=5, 
                               convert_unk=False, length_normalization_factor=None,
                               max_words=None, logprobs_history=False, alignment_history=False):
    scope = tf.get_variable_scope()
    if not FLAGS.showtell_noimage:
      with tf.variable_scope(FLAGS.showtell_encode_scope or scope):
        attention_states, initial_state, image_emb = self._encode(image)
    else:
      image_emb = tf.zeros([melt.get_batch_size(image), self.emb_dim])
      initial_state = None
      attention_states = None

    with tf.variable_scope(FLAGS.showtell_decode_scope or scope):
      # max_words = max_words or TEXT_MAX_WORDS
      max_words = max_words or FLAGS.decoder_max_words
      decode_func = None 
      if decode_method == SeqDecodeMethod.greedy:
        decode_func = self.decoder.generate_sequence_greedy
      elif decode_method == SeqDecodeMethod.multinomal:
        decode_func = self.decoder.generate_sequence_multinomial
      if decode_func is not None:
        results = decode_func(image_emb, 
                          max_words=max_words, 
                          initial_state=initial_state,
                          attention_states=attention_states,
                          convert_unk=convert_unk,
                          need_logprobs=FLAGS.greedy_decode_with_logprobs)
      else:
        if decode_method == SeqDecodeMethod.ingraph_beam:
          decode_func = self.decoder.generate_sequence_ingraph_beam
        elif decode_method == SeqDecodeMethod.outgraph_beam:
          decode_func = self.decoder.generate_sequence_outgraph_beam
        else:
          raise ValueError('not supported decode_method: %s' % decode_method)
        
        results = decode_func(image_emb, 
                          max_words=max_words, 
                          initial_state=initial_state,
                          beam_size=beam_size, 
                          convert_unk=convert_unk,
                          attention_states=attention_states,
                          length_normalization_factor=length_normalization_factor or FLAGS.length_normalization_factor,
                          logprobs_history=logprobs_history,
                          alignment_history=alignment_history)
      if logprobs_history:
        if self.decoder.log_probs_history is not None:
          tf.add_to_collection('decoder_logprobs_history', self.decoder.log_probs_history)
      if alignment_history:
        if self.decoder.alignment_history is not None:
          tf.add_to_collection('decoder_alignment_history', self.decoder.alignment_history)
    return results

  def build_predict_graph(self, image, text, exact_prob=False, exact_loss=False):
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
    
    loss = self.build_graph(image, text)
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score
  
  def predict(self, image, text):
    """
    default usage is one single image , single text predict one sim score
    """
    #hack for big feature problem, input is reading raw image...
    if FLAGS.pre_calc_image_feature and isinstance(image[0], (str, np.string_)):
      if self.image_model is None:
        #notice must not pass self.sess! will reload fail FIXME TODO
        self.image_model = melt.ImageModel(FLAGS.image_checkpoint_file, FLAGS.image_model_name, feature_name=FLAGS.image_endpoint_feature_name)
      #HACK here only used for show attend and tell
      image = self.image_model.gen_features(image)

    feed_dict = {
      self.image_feature_feed: image,
      self.text_feed: text,
    }
    score = self.sess.run(self.score, feed_dict)
    return score

  def predict_text(self, images):
    """
    for translation evaluation only
    """
    #hack for big feature problem, input is reading raw image...
    if FLAGS.pre_calc_image_feature and isinstance(images[0], (str, np.string_)):
      if self.image_model is None:
        self.image_model = melt.ImageModel(FLAGS.image_checkpoint_file, FLAGS.image_model_name, feature_name=FLAGS.image_endpoint_feature_name)
      #HACK here only used for show attend and tells
      images = self.image_model.gen_features(images)

    feed_dict = {
      self.image_feature_feed: images,
      }

    feed_dict = gezi.merge_dicts(feed_dict, self.feed_dict)

    ops = []
    #if not FLAGS.reinforcement_learning:
    assert self.beam_text is not None, 'Forget to set predictor.beam_text, predictor.beam_text_score = beam_text, beam_text_score ?'
    ops = [self.beam_text, self.beam_text_score]
    #else:
    #  ops = [tf.expand_dims(self.text, 1), tf.expand_dims(self.text_score, 1)]

    if self.logprobs_history:
      if self.decoder.log_probs_history is not None:
        ops += [self.decoder.log_probs_history]
    
    if self.alignment_history:
      #TODO FIXME why here will cause showattentell-finetune-alignloss.sh 
      #Invalid argument: You must feed a value for placeholder tensor 'show_and_tell/main/init_2/text' with dtype int64 and shape [?,100]
      if self.decoder.alignment_history is not None:
        ops += [self.decoder.alignment_history]
      
    return self.sess.run(ops, feed_dict=feed_dict)
