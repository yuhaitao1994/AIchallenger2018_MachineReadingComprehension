"""
AI Challenger观点型问题阅读理解

main.py：train and test

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import tensorflow as tf
import json as json
import numpy as np
from tqdm import tqdm
import os
import codecs
import time
import math

from model_addAnswer_newGraph import Model
from util_addAnswer import *


def train(config):
    """
    训练与验证函数
    """
    with open(config.id2vec_file, "r") as fh:
        id2vec = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    dev_total = 29968  # 验证集数据量

    # 不同参数的训练在不同的文件夹下存储
    log_dir = config.log_dir + config.experiment
    save_dir = config.save_dir + config.experiment
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)

    # 可馈送迭代器，通过feed_dict机制选择每次sess.run时调用train_iterator还是dev_iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    # 选取模型
    if config.model_name == "default":
        model = Model(config, iterator, id2vec)
    else:
        print("model error")
        return

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_learning_rate
    emb_lr = config.init_emb_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)  # 存储计算图
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.learning_rate,
                           tf.constant(lr, dtype=tf.float32)))
        if config.training_embedding:
            sess.run(tf.assign(model.emb_lr, tf.constant(
                emb_lr, dtype=tf.float32)))

        best_dev_acc = 0.0  # 定义一个最佳验证准确率，只有当准确率高于它才保存模型
        print("Training ...")
        for go in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                      handle: train_handle})
            if global_step % config.period == 0:  # 每隔一段步数就记录一次train_loss和learning_rate
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                lr_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/learning_rate", simple_value=sess.run(model.learning_rate)), ])
                writer.add_summary(lr_sum, global_step)
                if config.training_embedding:
                    emb_lr_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/emb_lr", simple_value=sess.run(model.emb_lr)), ])
                    writer.add_summary(emb_lr_sum, global_step)

            if global_step % config.checkpoint == 0:  # 验证acc，并保存模型
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                # 评估训练集
                _, summ = evaluate_batch(
                    model, config.val_num_batches, train_eval_file, sess, "train_eval", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)

                # 评估验证集
                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()  # 将事件文件刷新到磁盘

                # 1101
                if global_step <= 40000:
                    lr = config.init_learning_rate
                elif global_step <= 60000:
                    lr = config.init_learning_rate
                    emb_lr = 1e-5
                elif global_step <= 120000:
                    lr = config.init_learning_rate / \
                        math.sqrt((global_step - 50000) / 10000)
                    emb_lr = (1e-5) / \
                        math.sqrt((global_step - 50000) / 10000)
                elif global_step <= 200000:
                    lr = config.init_learning_rate / \
                        math.sqrt((global_step - 50000) / 5000)
                    emb_lr = (1e-5) / \
                        math.sqrt((global_step - 50000) / 5000)
                else:
                    lr = config.init_learning_rate / \
                        math.sqrt(global_step / 1000)
                    emb_lr = (1e-5) / \
                        math.sqrt(global_step / 1000)

                sess.run(tf.assign(model.learning_rate,
                                   tf.constant(lr, dtype=tf.float32)))
                if config.training_embedding:
                    sess.run(tf.assign(model.emb_lr, tf.constant(
                        emb_lr, dtype=tf.float32)))

                # 保存模型的逻辑
                if metrics["accuracy"] > best_dev_acc:
                    best_dev_acc = metrics["accuracy"]
                    filename = os.path.join(
                        save_dir, "model_{}_devAcc_{:.6f}.ckpt".format(global_step, best_dev_acc))
                    saver.save(sess, filename)

    print("finished!")


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    """
    模型评估函数
    """
    answer_dict = {}  # 答案词典
    truth_dict = {}  # 真实答案词典
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, truth, answer = sess.run(
            [model.qa_id, model.loss, model.answer, model.classes], feed_dict={handle: str_handle})
        answer_dict_ = {}
        truth_dict_ = {}
        for ids, tr, ans in zip(qa_id, truth, answer):
            answer_dict_[str(ids)] = ans
            truth_dict_[str(ids)] = tr
        answer_dict.update(answer_dict_)
        truth_dict.update(truth_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate_acc(truth_dict, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/accuracy".format(data_type), simple_value=metrics["accuracy"]), ])
    return metrics, [loss_sum, acc_sum]


def test(config):
    """
    测试函数
    """
    with open(config.id2vec_file, "r") as fh:
        id2vec = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)

    total = 10000
    # 读取模型的路径和预测存储的路径
    save_dir = config.save_dir + config.experiment
    if not os.path.exists(save_dir):
        print("no save!")
        return
    predic_time = time.strftime("%Y-%m-%d_%H:%M:%S ", time.localtime())
    prediction_file = os.path.join(
        config.prediction_dir, (predic_time + "_predictions.txt"))

    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config), config).make_one_shot_iterator()

    # 选取模型
    if config.model_name == "default":
        model = Model(config, test_batch, id2vec, trainable=False)
    else:
        print("model error")
        return

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    print("testing ...")
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        answer_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            # 预测答案
            qa_id, answer = sess.run([model.qa_id, model.classes])
            answer_dict_ = {}
            for ids, ans in zip(qa_id, answer):
                answer_dict_[str(ids)] = ans
            answer_dict.update(answer_dict_)
        # 将结果写文件的操作，不用考虑问题顺序
        if len(answer_dict) != len(test_eval_file):
            print("data number not match")
        predictions = []
        for key, value in answer_dict.items():
            prediction_answer = test_eval_file[str(key)][value]
            predictions.append(str(key) + '\t' + str(prediction_answer))
        outputs = u'\n'.join(predictions)
        with codecs.open(prediction_file, 'w', encoding='utf-8') as f:
            f.write(outputs)
        print("done!")
