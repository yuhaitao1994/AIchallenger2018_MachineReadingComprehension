"""
AI Challenger观点型问题阅读理解

focal_loss.py

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import tensorflow as tf


def sparse_focal_loss(logits, labels, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    with tf.name_scope("focal_loss"):
        y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
        labels = tf.one_hot(labels, depth=y_pred.shape[1])
        L = -labels * ((1 - y_pred)**gamma) * tf.log(y_pred)
        L = tf.reduce_sum(L, axis=1)
        return L

'''
if __name__ == '__main__':
    labels = tf.constant([0, 1], name="labels")
    logits = tf.constant([[0.7, 0.2, 0.1], [0.6, 0.1, 0.3]], name="logits")
    a = tf.reduce_mean(sparse_focal_loss(logits, tf.stop_gradient(labels)))
    with tf.Session() as sess:
        print(sess.run(a))'''
