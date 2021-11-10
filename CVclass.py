# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:38:52 2021

@author: LJ
"""

from tensorflow.keras import backend as K
#import tensorflow.compat.v1 as v1
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss

def eightway_activation(x):
  """Retrieves neighboring pixels/features on the eight corners from
  a 3x3 patch.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels, 8]
  """
  # Get the number of channels in the input.
  shape_x = x.get_shape().as_list()
  if len(shape_x) != 4:
    raise ValueError('Only support for 4-D tensors!')

  # Pad at the margin.
  x = tf.pad(x,
             paddings=[[0,0],[1,1],[1,1],[0,0]],
             mode='SYMMETRIC')
  # Get eight neighboring pixels/features.
  x_groups = [
    x[:, 1:-1, :-2, :], # left
    x[:, 1:-1, 2:, :], # right
    x[:, :-2, 1:-1, :], # up
    x[:, 2:, 1:-1, :], # down
    x[:, :-2, :-2, :], # left-up
    x[:, 2:, :-2, :], # left-down
    x[:, :-2, 2:, :], # right-up
    x[:, 2:, 2:, :] # right-down
  ]
  output = [
    tf.expand_dims(c, axis=-1) for c in x_groups
  ]
  output = tf.concat(output, axis=-1)

  return output


def eightcorner_activation(x, size):
  """Retrieves neighboring pixels one the eight corners from a
  (2*size+1)x(2*size+1) patch.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]
    size: A number indicating the half size of a patch.

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels, 8]
  """
  # Get the number of channels in the input.
  # shape_x = x.get_shape().as_list()
  # if len(shape_x) != 4:
  #   raise ValueError('Only support for 4-D tensors!')
  #n, h, w, c = shape_x
  
  n = K.shape(x)[0]
  h = K.shape(x)[1]
  w = K.shape(x)[2]
  c = K.shape(x)[3]
   
  # Pad at the margin.
  p = size
  x_pad = tf.pad(x,
                 paddings=[[0,0],[p,p],[p,p],[0,0]],
                 mode='CONSTANT',
                 constant_values=0)

  # Get eight corner pixels/features in the patch.
  x_groups = []
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        # Ignore the center pixel/feature.
        continue

      x_neighbor = x_pad[:, st_y:st_y+h, st_x:st_x+w, :]
      x_groups.append(x_neighbor)

  output = [tf.expand_dims(c, axis=-1) for c in x_groups]
  output = tf.concat(output, axis=-1)

  return output

#%%
def ignores_from_label(labels,size):
  """Retrieves ignorable pixels from the ground-truth labels.

  This function returns a binary map in which 1 denotes ignored pixels
  and 0 means not ignored ones. For those ignored pixels, they are not
  only the pixels with label value >= num_classes, but also the
  corresponding neighboring pixels, which are on the the eight cornerls
  from a (2*size+1)x(2*size+1) patch.
  
  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    num_classes: A number indicating the total number of valid classes. The 
      labels ranges from 0 to (num_classes-1), and any value >= num_classes
      would be ignored.
    size: A number indicating the half size of a patch.

  Return:
    A tensor of size [batch_size, height_in, width_in, 8]
  """
  # Get the number of channels in the input.
  # shape_lab = labels.get_shape().as_list()
  # if len(shape_lab) != 3:
  #   raise ValueError('Only support for 3-D label tensors!')
  n = K.shape(labels)[0]
  h = K.shape(labels)[1]
  w = K.shape(labels)[2]

  # Retrieve ignored pixels with label value >= num_classes.
  num_classes=2
  ignore = tf.greater(labels, num_classes-1) # NxHxW
  #ignore=(labels == -1)

  # Pad at the margin.
  p = size
  ignore_pad = tf.pad(ignore,
                      paddings=[[0,0],[p,p],[p,p]],
                      mode='CONSTANT',
                      constant_values=0)

  # Retrieve eight corner pixels from the center, where the center
  # is ignored. Note that it should be bi-directional. For example,
  # when computing AAF loss with top-left pixels, the ignored pixels
  # might be the center or the top-left ones.
  ignore_groups= []
  for st_y in range(2*size,-1,-size):
    for st_x in range(2*size,-1,-size):
      if st_y == size and st_x == size:
        continue
      ignore_neighbor = ignore_pad[:,st_y:st_y+h,st_x:st_x+w]
      mask = tf.logical_or(ignore_neighbor, ignore)
      ignore_groups.append(mask)

  ig = 0
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        continue
      ignore_neighbor = ignore_pad[:,st_y:st_y+h,st_x:st_x+w]
      mask = tf.logical_or(ignore_neighbor, ignore_groups[ig])
      ignore_groups[ig] = mask
      ig += 1

  ignore_groups = [
    tf.expand_dims(c, axis=-1) for c in ignore_groups
  ] # NxHxWx1
  ignore = tf.concat(ignore_groups, axis=-1) #NxHxWx8

  return ignore

#%%
def edges_from_label(labels, size, ignore_class=255):
  """Retrieves edge positions from the ground-truth labels.

  This function computes the edge map by considering if the pixel values
  are equal between the center and the neighboring pixels on the eight
  corners from a (2*size+1)*(2*size+1) patch. Ignore edges where the any
  of the paired pixels with label value >= num_classes.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    size: A number indicating the half size of a patch.
    ignore_class: A number indicating the label value to ignore.

  Return:
    A tensor of size [batch_size, height_in, width_in, 1, 8]
  """
  # Get the number of channels in the input.
  # shape_lab = labels.get_shape().as_list()
  # if len(shape_lab) != 4:
  #   raise ValueError('Only support for 4-D label tensors!')
  
  n = K.shape(labels)[0]
  h = K.shape(labels)[1]
  w = K.shape(labels)[2]
  c = K.shape(labels)[3]

  # Pad at the margin.
  p = size
  labels_pad = tf.pad(
    labels, paddings=[[0,0],[p,p],[p,p],[0,0]],
    mode='CONSTANT',
    constant_values=ignore_class)

  # Get the edge by comparing label value of the center and it paired pixels.
  edge_groups= []
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        continue
      labels_neighbor = labels_pad[:,st_y:st_y+h,st_x:st_x+w]
      edge = tf.not_equal(labels_neighbor, labels)
      edge_groups.append(edge)

  edge_groups = [
    tf.expand_dims(c, axis=-1) for c in edge_groups
  ] # NxHxWx1x1
  edge = tf.concat(edge_groups, axis=-1) #NxHxWx1x8

  return edge

class ExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, dec, epoch_nr):
        self.dec = dec
        self.epoch_nr = epoch_nr

   
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        decrement = tf.math.pow(10.0, -epoch/self.epoch_nr)
        K.set_value(self.dec, decrement)


    
class AAF_Loss(Loss):
    """
    Loss function for multiple outputs
    """
    def __init__(self, alpha, EPOCH_NR, ignore_index=255, num_classes=2):
        super(AAF_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.kld_margin = 3.0
        self.kld_lambda_1 = 1.0
        self.kld_lambda_2 = 1.0
        self.dec=alpha
        self.EPOCH_NR=EPOCH_NR
        self.w_edge = tf.nn.softmax(tf.zeros(shape=(1, 1, 1, self.num_classes, 1, 3),dtype=tf.dtypes.float32, name='w_edge'),axis=-1,name=None)    
        self.w_not_edge =tf.nn.softmax(tf.zeros(shape=(1, 1, 1, self.num_classes, 1, 3),dtype=tf.dtypes.float32, name='w_not_edge'),axis=-1,name=None) 

    def call(self, labels, prob):
        h = K.shape(labels)[1]
        w = K.shape(labels)[2]

#%%    
       # w_edge = tf.nn.softmax(self.w_edge, axis=-1, name=None)
        #w_not_edge = tf.nn.softmax(self.w_not_edge, axis=-1, name=None)
        prob = tf.nn.softmax(prob, axis=-1)

        # Apply AAF on 3x3 patch.
        eloss_1, neloss_1= adaptive_affinity_loss(labels,
                                                          prob,
                                                          1,
                                                          self.num_classes,
                                                          self.kld_margin,
                                                          self.w_edge[..., 0],
                                                          self.w_not_edge[..., 0])
        # #Apply AAF on 5x5 patch.
        # eloss_2, neloss_2 = adaptive_affinity_loss(labels,
        #                                                   prob,
        #                                                   2,
        #                                                   self.num_classes,
        #                                                   self.kld_margin,
        #                                                   w_edge[..., 1],
        #                                                   w_not_edge[..., 1])
        # Apply AAF on 7x7 patch.
        # eloss_3, neloss_3 = adaptive_affinity_loss(labels,
        #                                                   prob,
        #                                                   3,
        #                                                   self.num_classes,
        #                                                   self.kld_margin,
        #                                                   self.w_edge[..., 2],
        #                                                   self.w_not_edge[..., 2])

        dec=self.dec
        aaf_loss = tf.reduce_mean(eloss_1) * self.kld_lambda_1 * dec
        # aaf_loss += tf.reduce_mean(eloss_2) * self.kld_lambda_1*dec
         #aaf_loss = tf.reduce_mean(eloss_3) * self.kld_lambda_1*dec
        aaf_loss += tf.reduce_mean(neloss_1) * self.kld_lambda_2*dec
        # aaf_loss += tf.reduce_mean(neloss_2) * self.kld_lambda_2*dec
        #aaf_loss += tf.reduce_mean(neloss_3) * self.kld_lambda_2*dec
        #cross_entropy = -labels * K.log(prob)
     #   normal_loss=  K.mean(K.sum(cross_entropy, axis=[-1]))
      #  final_loss = normal_loss + aaf_loss
        return aaf_loss
    
def Dice_and_AAF_v1():
    def combination_loss(y_true, y_pred):
        return dice_coefficient(y_true, y_pred) + AAF_Loss(y_true, y_pred)
    return combination_loss


def focal_tversky_loss_and_AAF(alpha=0.5,EPOCH_NR=600):
    def loss_function(y_true,y_pred):
        dice = focal_tversky_loss(delta=0.1)(y_true, y_pred)
        AAF=AAF_Loss(alpha,EPOCH_NR)(y_true, y_pred)
        combo_loss= dice+ AAF
        return combo_loss 
    return loss_function

#%%
def adaptive_affinity_loss(labels,
                           probs,
                           size,
                           num_classes,
                           kld_margin,
                           w_edge,
                           w_not_edge):
   # def loss_function(y_true, probs):   
     #Compute ignore map (e.g, label of 255 and their paired pixels).
     #IMAGES ARE UINT8 FROM 0 TO 255 
     real_label=labels
     one_channel = real_label[:,:,:,0]+real_label[:,:,:,1]
     y_true=one_channel*255
     y_true=tf.cast(y_true, tf.uint8)
     labels=y_true
    # labels = tf.math.reduce_max(y_true, axis=-1) # NxHxW
     #labels=tf.cast(labels, tf.int32)
     one_hot_lab = tf.one_hot(labels, depth=num_classes) #needs 
     ignore = ignores_from_label(labels, size) # NxHxWx8 #num_classes=2, size=3
     not_ignore = tf.logical_not(ignore)
     not_ignore = tf.expand_dims(not_ignore, axis=3) # NxHxWx1x8
       # Compute edge map.
     edge = edges_from_label(one_hot_lab, size, 255) # NxHxWxCx8
        
       # Remove ignored pixels from the edge/non-edge.
     edge = tf.logical_and(edge, not_ignore)
     not_edge = tf.logical_and(tf.logical_not(edge), not_ignore)
        
     edge_indices = tf.where(tf.reshape(edge, [-1]))
     not_edge_indices = tf.where(tf.reshape(not_edge, [-1]))
    # Extract eight corner from the center in a patch as paired pixels.
     probs_paired = eightcorner_activation(probs, size)  # NxHxWxCx8
    # probs_paired=tf.cast(probs_paired, tf.float32)

    # probsx = tf.cast(probs, "float32");
     probsx = tf.expand_dims(probs, axis=-1) # NxHxWxCx1
     bot_epsilon = tf.constant(1e-4, name='bot_epsilon')
     top_epsilon = tf.constant(1.0, name='top_epsilon')
     
     neg_probs = tf.clip_by_value(
            1-probsx, bot_epsilon, top_epsilon)
     neg_probs_paired = tf.clip_by_value(
            1-probs_paired, bot_epsilon, top_epsilon)
     probsx = tf.clip_by_value(
           probsx, bot_epsilon, top_epsilon)
     probs_paired = tf.clip_by_value(
          probs_paired, bot_epsilon, top_epsilon)
     
       # Compute KL-Divergence.
     kldiv = probs_paired*tf.math.log(probs_paired/probsx)
     kldiv += neg_probs_paired*tf.math.log(neg_probs_paired/neg_probs) #+=
 
     edge_loss = tf.maximum(0.0, kld_margin-kldiv)
     not_edge_loss = kldiv
     
     w_edge=tf.convert_to_tensor(w_edge,tf.float32)
     w_not_edge=tf.convert_to_tensor(w_not_edge,tf.float32)
     
        # Impose weights on edge/non-edge losses.
     one_hot_lab = tf.expand_dims(one_hot_lab, axis=-1)
     w_edge = tf.reduce_sum(w_edge*one_hot_lab, axis=3, keepdims=True) # NxHxWx1x1
     w_not_edge = tf.reduce_sum(w_not_edge*one_hot_lab, axis=3, keepdims=True) # NxHxWx1x1
        
     edge_loss *= w_edge
     not_edge_loss *= w_not_edge
     
     not_edge_loss = tf.reshape(not_edge_loss, [-1])
     not_edge_loss = tf.gather(not_edge_loss, not_edge_indices)
     edge_loss = tf.reshape(edge_loss, [-1])
     edge_loss = tf.gather(edge_loss, edge_indices)

     return edge_loss, not_edge_loss
 
 ################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_tversky_loss = K.sum(K.pow((1-tversky_class), gamma), axis=[-1])
    	# adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        focal_tversky_loss = focal_tversky_loss / num_classes
        return focal_tversky_loss

# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')