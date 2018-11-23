"""
AI Challenger观点型问题阅读理解

examine_dev.py：检查dev集的结果，辅助分析

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import tensorflow as tf
import json as json
import numpy as np
from tqdm import tqdm
import pickle
import os
import codecs
import time
from model import Model
from util import *


def save_dev(config):
    """
    验证dev集的结果，保存文件
    """
    with open(config.id2vec_file, "r") as fh:
        id2vec = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    total = 29968

    print("Loading model...")
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    truth_dict = {}
    predict_dict = {}
    logits_dict1 = {}

    print("正在模型预测!")
    g1 = tf.Graph()
    with tf.Session(graph=g1, config=sess_config) as sess1:
        with g1.as_default():
            dev_batch1 = get_dataset(config.dev_record_file, get_record_parser(
                config), config).make_one_shot_iterator()
            model_1 = Model(config, dev_batch1, id2vec, trainable=False)
            sess1.run(tf.global_variables_initializer())
            saver1 = tf.train.Saver()
            # 需要手动更改路径
            saver1.restore(
                sess1, "./log/model/model_10000_devAcc_0.662240.ckpt")
            sess1.run(tf.assign(model_1.is_train,
                                tf.constant(False, dtype=tf.bool)))
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, logits, truths = sess1.run(
                    [model_1.qa_id, model_1.logits, model_1.answer])
                for ids, logits, truth in zip(qa_id, logits, truths):
                    logits_dict1[str(ids)] = logits
                    truth_dict[str(ids)] = truth
            if len(logits_dict1) != len(dev_eval_file):
                print("logits1 data number not match")

            a = tf.placeholder(shape=[3], dtype=tf.float32, name="me")
            softmax = tf.nn.softmax(a)
            for key, val in truth_dict.items():
                value = sess1.run(softmax, feed_dict={a: logits_dict1[key]})
                predict_dict[key] = value
    print("正在保存dev的softmax结果文件!")
    if not os.path.exists("./dev_soft"):
        os.makedirs("./dev_soft")
    with open("./dev_soft/BIDAF_b64_e256_h150_v256.txt", "wb") as f1:  # 手动更改保存的名字，路径不用改
        pickle.dump(predict_dict, f1)
    if not os.path.exists("./truth"):
        os.makedirs("./truth")
    with open("./truth/truth_dict.txt", "wb") as f2:  # 不用改
        pickle.dump(truth_dict, f2)


def save_test(config):
    """
    输出test集的结果，保存文件
    """
    with open(config.id2vec_file, "r") as fh:
        id2vec = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)
    total = 10000

    print("Loading model...")
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    predict_dict = {}
    logits_dict1 = {}

    print("正在模型预测!")
    g1 = tf.Graph()
    with tf.Session(graph=g1, config=sess_config) as sess1:
        with g1.as_default():
            test_batch1 = get_dataset(config.test_record_file, get_record_parser(
                config), config).make_one_shot_iterator()
            model_1 = Model(config, test_batch1, id2vec, trainable=False)
            sess1.run(tf.global_variables_initializer())
            saver1 = tf.train.Saver()
            # 需要手动更改路径
            saver1.restore(
                sess1, "./log/model/model_131000_devAcc_0.732782.ckpt")
            sess1.run(tf.assign(model_1.is_train,
                                tf.constant(False, dtype=tf.bool)))
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, logits = sess1.run(
                    [model_1.qa_id, model_1.logits])
                for ids, logits in zip(qa_id, logits):
                    logits_dict1[str(ids)] = logits
            if len(logits_dict1) != len(test_eval_file):
                print("logits1 data number not match")

            a = tf.placeholder(shape=[3], dtype=tf.float32, name="me")
            softmax = tf.nn.softmax(a)
            for key, val in logits_dict1.items():
                value = sess1.run(softmax, feed_dict={a: logits_dict1[key]})
                predict_dict[key] = value

    print("正在保存dev的softmax结果文件!")
    if not os.path.exists("./test_soft"):
        os.makedirs("./test_soft")
    with open("./test_soft/RNET_b64_e300_h60_v300.txt", "wb") as f1:
        pickle.dump(predict_dict, f1)
