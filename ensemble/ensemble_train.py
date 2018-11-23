"""
AI Challenger观点型问题阅读理解

ensemble_train.py:在验证集中训练模型融合的权重。

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensemble_train():
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
    for i in range(len(filenames)):
        print("{}: {}".format(i + 1, filenames[i]))
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
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # run
        sess.run(tf.global_variables_initializer())
        best_acc = 0.
        best_wb = {}
        for steps in range(1500):
            answers, train_OP = sess.run([predictions, train_op], feed_dict={
                                         inputs: all_inputs, labels: all_labels})
            answer_dict = {}
            for i in range(total):
                answer_dict[keys[i]] = answers[i]
            if evaluate_acc(truth_dict, answer_dict)["accuracy"] > best_acc:
                best_acc = evaluate_acc(truth_dict, answer_dict)["accuracy"]
                best_wb["weights"] = sess.run(W).tolist()
                best_wb["bias"] = sess.run(b).tolist()
            if (steps + 1) % 50 == 0:
                print("steps:{},acc:{:.5f}".format(
                    steps + 1, evaluate_acc(truth_dict, answer_dict)["accuracy"]))
        for i in range(len(best_wb["weights"])):
            print("{}: {}".format(i + 1, best_wb["weights"][i]))
        save_wb = {}
        for file, weight in zip(filenames, best_wb["weights"]):
            save_wb[file] = weight
        save_wb["bias"] = best_wb["bias"]
        print(best_acc)
        with open("ensemble_wb.json", "w") as fw:
            json.dump(save_wb, fw)


def evaluate_acc(truth_dict, answer_dict):
    """
    计算准确率，还可以设计返回正确问题和错误问题列表
    """
    total = 0
    right = 0
    wrong = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = truth_dict[key]
        prediction = value
        if prediction == ground_truths:
            right += 1
        else:
            wrong += 1
    accuracy = (right / total) * 1.0
    return {"accuracy": accuracy}


if __name__ == '__main__':
    ensemble_train()
