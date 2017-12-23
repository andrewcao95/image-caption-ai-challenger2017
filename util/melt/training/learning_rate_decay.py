#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   learning_rate_decay.py
#        \author   chenghuige  
#          \date   2017-10-21 06:11:06.077589
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

def piecewise_constant(x, boundaries, values, name=None):
  with ops.name_scope(name, "PiecewiseConstant",
                      [x, boundaries, values, name]) as name:
    x = ops.convert_to_tensor(x)  
    #TODO hack chg change this from traing/learning_rate_decay.py
    # Avoid explicit conversion to x's dtype. This could result in faulty
    # comparisons, for example if floats are converted to integers.
    boundaries = ops.convert_n_to_tensor(boundaries, tf.int64)
    for b in boundaries:
      if b.dtype.base_dtype != x.dtype.base_dtype:
        raise ValueError(
            "Boundaries (%s) must have the same dtype as x (%s)." % (
                b.dtype.base_dtype, x.dtype.base_dtype))
    # TODO(rdipietro): Ensure that boundaries' elements are strictly increasing.
    values = ops.convert_n_to_tensor(values)
    for v in values[1:]:
      if v.dtype.base_dtype != values[0].dtype.base_dtype:
        raise ValueError(
            "Values must have elements all with the same dtype (%s vs %s)." % (
                values[0].dtype.base_dtype, v.dtype.base_dtype))

    pred_fn_pairs = {}
    pred_fn_pairs[x <= boundaries[0]] = lambda: values[0]
    pred_fn_pairs[x > boundaries[-1]] = lambda: values[-1]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
      # Need to bind v here; can do this with lambda v=v: ...
      pred = (x > low) & (x <= high)
      pred_fn_pairs[pred] = lambda v=v: v

    # The default isn't needed here because our conditions are mutually
    # exclusive and exhaustive, but tf.case requires it.
    default = lambda: values[0]
    return control_flow_ops.case(pred_fn_pairs, default, exclusive=True) 
