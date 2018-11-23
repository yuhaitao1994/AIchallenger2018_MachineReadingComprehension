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


def ensemble_train(config):
    total = 29968
    print("正在读取dev的softmax结果文件!")
    rootdir = "./dev_soft"
    dev_dict = {}
    predict_dict = {}
    with open('truth/truth_dict.txt', 'rb') as f1:
        truth_dict = pickle.load(f1)
    filelist = os.listdir(rootdir)
    filenames = [
        filename for filename in filelist if not filename.startswith('.')]
    print(filenames)
    if len(filenames) == 0:
        print("没有softmax文件")
        return

    # 定义整个输入和标签矩阵
    all_inputs = np.zeros(shape=[total, 3, len(filenames)], dtype=np.float32)
    all_labels = np.zeros(shape=[total], dtype=np.int32)
    keys = []
    for k in truth_dict.keys():
        keys.append(k)
    keys.sort(reverse=False)
    if len(keys) != total:
        print("keys number error")
        return
    # 给标签赋值
    for i in range(total):
        all_labels[i] = truth_dict[keys[i]]
    # 遍历文件加入矩阵
    for i in range(len(filenames)):
        path = os.path.join(rootdir, filenames[i])
        with open(path, "rb") as f1:
            soft = pickle.load(f1)
            for j in range(total):
                all_inputs[j, :, i] = soft[keys[j]]
            print(i + 1, "个文件处理完成")
    # print(all_labels[:10])
    # print(all_inputs[:10, :, :])
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        inputs = tf.placeholder(shape=[total, 3, len(
            filenames)], dtype=tf.float32, name="inputs")
        labels = tf.placeholder(shape=[total], dtype=tf.int32, name="labels")
        W = tf.get_variable(shape=[len(filenames), 1],
                            dtype=tf.float32, name="weights")
        b = tf.get_variable(
            shape=[1], dtype=tf.float32, name="bias")
        re_inputs = tf.reshape(inputs, shape=[-1, len(filenames)])
        pred = tf.matmul(re_inputs, W)
        re_pred = tf.reshape(pred, shape=[total, 3, 1])
        outputs = tf.squeeze(re_pred)
        predictions = tf.cast(tf.argmax(outputs, axis=1), tf.int32)
        # loss and opt
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs, labels=tf.stop_gradient(labels)))
        train_op = tf.train.GradientDescentOptimizer(0.7).minimize(loss)

        # run
        sess.run(tf.global_variables_initializer())
        best_acc = 0.
        best_wb = {}
        for steps in range(800):
            answers, train_OP = sess.run([predictions, train_op], feed_dict={
                                         inputs: all_inputs, labels: all_labels})
            answer_dict = {}
            for i in range(total):
                answer_dict[keys[i]] = answers[i]
            if evaluate_acc(truth_dict, answer_dict)["accuracy"] > best_acc:
                best_acc = evaluate_acc(truth_dict, answer_dict)["accuracy"]
                best_wb["weights"] = sess.run(W).tolist()
                best_wb["bias"] = sess.run(b).tolist()
            if (steps + 1) % 40 == 0:
                print("steps:{},acc:{:.5f}".format(
                    steps + 1, evaluate_acc(truth_dict, answer_dict)["accuracy"]))
        print(best_acc)
        print(best_wb)

        with open("ensemble_wb.json", "w") as fw:
            json.dump(best_wb, fw)


def load_devsoft():
    print("正在读取dev的softmax结果文件!")
    rootdir = "./dev_soft"
    # 定义融合后的验证集softmax字典
    dev_dict = {}
    predict_dict = {}
    # 读取正确答案文件truth_dict
    with open('truth/truth_dict.txt', 'rb') as f1:
        truth_dict = pickle.load(f1)

    # 获取目录下所有文件，并去除隐藏文件
    filelist = os.listdir(rootdir)
    filenames = [
        filename for filename in filelist if not filename.startswith('.')]
    print(filenames)

    # 定义权值，根据需要手动更改，依次对应soft_dev下的文件
    weights = [1.0] * 10

    # 初始化dev_dict
    if len(filenames) == 0:
        print("没有softmax文件")
        return
    path = os.path.join(rootdir, filenames[0])
    with open(path, "rb") as f1:
        soft = pickle.load(f1)
    for key, value in soft.items():
        dev_dict[key] = weights[0] * value
    print("初始化完成")

    # 遍历剩下的dev文件
    for i in range(1, len(filenames)):
        weight = weights[i]
        path = os.path.join(rootdir, filenames[i])
        with open(path, "rb") as f1:
            soft = pickle.load(f1)
            for key in soft.keys():
                dev_dict[key] += weight * soft[key]
        print(i + 1, "个文件处理完成")
    # 计算准确率
    with tf.Session(graph=tf.Graph()) as sess:
        ddev = tf.placeholder(shape=[3], dtype=tf.float32, name='all')
        dev_class = tf.cast(tf.argmax(ddev), dtype=tf.int32)
        for key, val in truth_dict.items():
            value = sess.run(dev_class, feed_dict={ddev: dev_dict[key]})
            predict_dict[key] = value
    print(evaluate_acc(truth_dict, predict_dict))


def load_testsoft(config):
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)
    with open("ensemble_wb.json", "r") as fh:
        best_wb = json.load(fh)

    predic_time = time.strftime("%Y-%m-%d_%H:%M:%S ", time.localtime())
    prediction_file = os.path.join(
        config.prediction_dir, (predic_time + "_predictions.txt"))

    print("正在读取test的softmax结果文件!")
    rootdir = "./test_soft"
    # 定义融合后的验证集softmax字典
    dev_dict = {}
    predict_dict = {}

    # 获取目录下所有文件，并去除隐藏文件
    filelist = os.listdir(rootdir)
    # filenames = [
    #     filename for filename in filelist if not filename.startswith('.')]

    print(filenames)

    # 初始化dev_dict
    if len(filenames) == 0:
        print("没有softmax文件")
        return
    path = os.path.join(rootdir, filenames[0])
    with open(path, "rb") as f1:
        soft = pickle.load(f1)
    for key, value in soft.items():
        dev_dict[key] = best_wb["weights"][0][0] * value
    print("初始化完成")

    # 遍历剩下的dev文件
    for i in range(1, len(filenames)):
        weight = best_wb["weights"][i][0]
        print(weight)
        path = os.path.join(rootdir, filenames[i])
        with open(path, "rb") as f1:
            soft = pickle.load(f1)
            for key in soft.keys():
                dev_dict[key] += weight * soft[key]
        print(i + 1, "个文件处理完成")
    for key in dev_dict.keys():
        dev_dict[key] += best_wb["bias"][0]
    # 计算准确率
    with tf.Session(graph=tf.Graph()) as sess:
        ddev = tf.placeholder(shape=[3], dtype=tf.float32, name='all')
        dev_class = tf.cast(tf.argmax(ddev), dtype=tf.int32)
        for k in range(280001, 290001):
            key = str(k)
            value = sess.run(dev_class, feed_dict={ddev: dev_dict[key]})
            predict_dict[key] = value

    predictions = []
    for key, value in predict_dict.items():
        prediction_answer = test_eval_file[str(key)][value]
        predictions.append(str(key) + '\t' + str(prediction_answer))
    outputs = u'\n'.join(predictions)
    with codecs.open(prediction_file, 'w', encoding='utf-8') as f:
        f.write(outputs)
    print("done!")
