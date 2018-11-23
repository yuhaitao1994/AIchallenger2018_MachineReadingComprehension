"""
AI Challenger观点型问题阅读理解

ensemble_predict.py：将模型权重用于融合，预测结果。

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

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def emsemble_predict():
    with open("test_eval.json", "r") as fh:
        test_eval_file = json.load(fh)
    with open("ensemble_wb.json", "r") as fh:
        best_wb = json.load(fh)

    predic_time = time.strftime("%Y-%m-%d_%H:%M:%S ", time.localtime())
    prediction_file = os.path.join(
        "predictions", (predic_time + "_predictions.txt"))

    print("正在读取test的softmax结果文件!")
    rootdir = "./test_soft"
    # 定义融合后的验证集softmax字典
    dev_dict = {}
    predict_dict = {}

    # 获取目录下所有文件，并去除隐藏文件
    filelist = os.listdir(rootdir)
    filenames = [
        filename for filename in filelist if not filename.startswith('.')]
    for i in range(len(filenames)):
        print("{}: {}".format(i + 1, filenames[i]))
#     print(filenames)

    # 初始化dev_dict
    if len(filenames) == 0:
        print("没有softmax文件")
        return
    path = os.path.join(rootdir, filenames[0])
    with open(path, "rb") as f1:
        soft = pickle.load(f1)
    for key, value in soft.items():
        dev_dict[key] = best_wb[filenames[0]][0] * value
    print("初始化完成")

    # 遍历剩下的dev文件
    for i in range(1, len(filenames)):
        weight = best_wb[filenames[i]][0]
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


if __name__ == '__main__':
    ensemble_predict()
