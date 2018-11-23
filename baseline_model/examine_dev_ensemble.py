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
import os
import pickle
import codecs
import time
from collections import Counter
from model import Model
from util import *


def examine_dev_ensemble(config):
    """
    检查dev集的结果，辅助分析
    """
    with open(config.id2vec_file, "r") as fh:
        id2vec = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    total = 29968 * 3
    # 读取模型的路径和预测存储的路径
    save_dir = config.save_dir + config.experiment
    if not os.path.exists(save_dir):
        print("no save!")
        return
    predic_time = time.strftime("%Y-%m-%d_%H:%M:%S ", time.localtime())
    os.path.join(config.prediction_dir, (predic_time + "_examine_dev.txt"))

    print("Loading model...")
    examine_batch = get_dataset("./data/dev_aug.tfrecords", get_record_parser(
        config), config).make_one_shot_iterator()

    model = Model(config, examine_batch, id2vec, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    print("examining ...")
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        logits_dict = {}
        truth_dict = {}
        answer_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            # 预测答案
            qa_id, softmax, truth = sess.run(
                [model.qa_id, model.final_softmax, model.answer])
            # 往字典中添加每个id的三个logits
            for ids, tr, log in zip(qa_id, truth, softmax):
                if str(ids) not in logits_dict.keys():
                    logits_dict[str(ids)] = log
                    truth_dict[str(ids)] = tr
                else:
                    logits_dict[str(ids)] += log
                # if str(ids) not in class_dict.keys():
                #     class_dict[str(ids)] = [int(cla)]
                #     truth_dict[str(ids)] = tr
                # else:
                #     class_dict[str(ids)].append(int(cla))
        # 根据合并的logits求answer
        for key, value in logits_dict.items():
            val = value.tolist()
            answer_dict[key] = val.index(max(val))
            # answer_dict[key], _ = Counter(value).most_common(1)[0]
        metrics = evaluate_acc(truth_dict, answer_dict)
        print(len(truth_dict))
        print(len(answer_dict))
        print("accuracy:{}".format(metrics["accuracy"]))

        print("正在保存dev的softmax结果文件!")
        if not os.path.exists("./dev_soft"):
            os.makedirs("./dev_soft")
        with open("./dev_soft/model_aug_dev_ensemble.txt", "wb") as f1:  # 手动更改保存的名字，路径不用改
            pickle.dump(logits_dict, f1)

        yes_predictions = []  # 正确答案是肯定的错题
        no_predictions = []  # 正确答案是否定的错题
        depend_predictions = []  # 正确答案是不确定的错题
        yes, no, depend = 0, 0, 0
        yes_wrong, no_wrong, depend_wrong = 0, 0, 0
        for key, value in answer_dict.items():
            if truth_dict[key] != value:
                if truth_dict[key] == 0:
                    yes += 1
                    yes_wrong += 1
                    right_answer = dev_eval_file[str(key)][truth_dict[key]]
                    wrong_answer = dev_eval_file[str(key)][value]
                    yes_predictions.append(
                        str(key) + '\t' + str(right_answer) + '\t' + str(wrong_answer))
                elif truth_dict[key] == 1:
                    no += 1
                    no_wrong += 1
                    right_answer = dev_eval_file[str(key)][truth_dict[key]]
                    wrong_answer = dev_eval_file[str(key)][value]
                    no_predictions.append(
                        str(key) + '\t' + str(right_answer) + '\t' + str(wrong_answer))
                else:
                    depend += 1
                    depend_wrong += 1
                    right_answer = dev_eval_file[str(key)][truth_dict[key]]
                    wrong_answer = dev_eval_file[str(key)][value]
                    depend_predictions.append(
                        str(key) + '\t' + str(right_answer) + '\t' + str(wrong_answer))
            else:
                if truth_dict[key] == 0:
                    yes += 1
                elif truth_dict[key] == 1:
                    no += 1
                else:
                    depend += 1

        print("肯定型问题个数:{},否定型问题个数:{},不确定问题个数:{}".format(yes, no, depend))
        print("肯定型问题正确率:{}".format((yes - yes_wrong) / yes * 1.0))
        print("否定型问题正确率:{}".format((no - no_wrong) / no * 1.0))
        print("不确定型问题正确率:{}".format((depend - depend_wrong) / depend * 1.0))
        outputs_0 = u'\n'.join(yes_predictions)
        outputs_1 = u'\n'.join(no_predictions)
        outputs_2 = u'\n'.join(depend_predictions)
        with codecs.open(os.path.join(config.prediction_dir, (predic_time + "_examine_dev_0.txt")), 'w', encoding='utf-8') as f:
            f.write(outputs_0)
        with codecs.open(os.path.join(config.prediction_dir, (predic_time + "_examine_dev_1.txt")), 'w', encoding='utf-8') as f:
            f.write(outputs_1)
        with codecs.open(os.path.join(config.prediction_dir, (predic_time + "_examine_dev_2.txt")), 'w', encoding='utf-8') as f:
            f.write(outputs_2)
        print("done!")
