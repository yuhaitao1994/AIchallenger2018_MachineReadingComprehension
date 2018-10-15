"""
AI Challenger观点型问题阅读理解

DemoModel.py：模型演示代码，测试模型能否跑通

@author: yuhaitao
"""
# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import random
from nn_func import cudnn_gru, native_gru, dot_attention, summ, ptr_net, dropout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow的log显示级别
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

hidden = 75
use_cudnn = False
batch_size = 2
learning_rate = 0.00001
keep_prob = 0.7
grad_clip = 5.0
len_limit = 15

class DemoModel(object):

    def __init__(self, word_mat, trainable=True, opt=True):
        # 注意，placeholder是数据传输的入口，不能在计算图中重新赋值
        self.passage = tf.placeholder(tf.int32, [batch_size, None], name="passage")
        self.question = tf.placeholder(tf.int32, [batch_size, None], name="question")
        self.answer = tf.placeholder(tf.int32, [batch_size], name="answer")
        self.qa_id = tf.placeholder(tf.int32, [batch_size], name="qa_id")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)

        self.c_mask = tf.cast(self.passage, tf.bool)
        self.q_mask = tf.cast(self.question, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.passage, [0, 0], [batch_size, self.c_maxlen])
            self.q = tf.slice(self.question, [0, 0], [batch_size, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [
                                   batch_size, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [
                                   batch_size, self.q_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.RNet()

        if trainable:
            self.learning_rate = tf.get_variable(
                "learning_rate", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def RNet(self):
        PL, QL, d = self.c_maxlen, self.q_maxlen, hidden
        gru = cudnn_gru if use_cudnn else native_gru

        with tf.variable_scope("embedding"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=batch_size, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(inputs=c, memory=q, mask=self.q_mask, hidden=d,
                                   keep_prob=keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)
            print(att.get_shape().as_list())

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=keep_prob, is_train=self.is_train)
            # match:[batch_size, c_maxlen, 6*hidden]
            match = rnn(self_att, seq_len=self.c_len)
            print(match.get_shape().as_list())

        with tf.variable_scope("YesNo_classification"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=keep_prob, is_train=self.is_train)
            print(init.get_shape().as_list())
            match = dropout(match, keep_prob=keep_prob,
                            is_train=self.is_train)
            final_hiddens = init.get_shape().as_list()[-1]
            final_gru = tf.contrib.rnn.GRUCell(final_hiddens)
            _, final_state = tf.nn.dynamic_rnn(
                final_gru, match, initial_state=init, dtype=tf.float32)
            final_w = tf.get_variable(name="final_w", shape=[final_hiddens, 2])
            final_b = tf.get_variable(name="final_b", shape=[
                                      2], initializer=tf.constant_initializer(0.))
            self.logits = tf.matmul(final_state, final_w)
            self.logits = tf.nn.bias_add(self.logits, final_b)  # logits:[batch_size, 3]

        with tf.variable_scope("softmax_and_loss"):
            final_softmax = tf.nn.softmax(self.logits)
            self.classes = tf.cast(
                tf.argmax(final_softmax, axis=1), dtype=tf.int32, name="classes")
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=tf.stop_gradient(self.answer)))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

def get_bacth(examples, word2idx_dict, batch_size):
    """
    获取mini-batch
    """
    passages = []
    questions = []
    answers = []
    qa_ids = []
    for i in range(batch_size):
        passage_id = []
        question_id = []
        passage = examples[i]["passage"]
        for j in range(15):
            if (j + 1) <= len(passage):
                p = passage[j]
                passage_id.append(word2idx_dict[p])
            else:
                passage_id.append(0)
        question = examples[i]["question"]
        for j in range(15):
            if (j + 1) <= len(question):
                q = question[j]
                question_id.append(word2idx_dict[q])
            else:
                question_id.append(0)
        answer = examples[i]["answer"]
        qa_id = examples[i]["qa_id"]
        passages.append(passage_id)
        questions.append(question_id)
        answers.append(answer)
        qa_ids.append(qa_id)
    passages = np.array(passages).astype(np.int32)
    questions = np.array(questions)
    answers = np.array(answers)
    qa_ids = np.array(qa_ids)
    return passages, questions, answers, qa_ids


def main(_):
    """
    测试模型的demo
    """
    train_examples = [
        {
            "passage": ['苹果', '是', '甜的', '它', '是', '硬的'],
            "question":['苹果', '是', '甜的', '吗'],
            "answer":0,
            "qa_id":1},
        {
            "passage": ['橘子', '是', '酸的', '它', '是', '软的', '也是', '好吃的'],
            "question":['橘子', '是', '甜的', '吗'],
            "answer":1,
            "qa_id":2},
        {
            "passage": ['梨', '是', '甜的', '它', '是', '硬的'],
            "question":['梨', '是', '软的', '吗'],
            "answer":1,
            "qa_id":3},
        {
            "passage": ['西瓜', '是', '甜的', '它', '是', '硬的', '也是', '大的', '和', '圆的'],
            "question":['西瓜', '是', '酸的', '吗'],
            "answer":2,
            "qa_id":4}
    ]

    dev_examples = [
        {
            "passage": ['葡萄', '是', '甜的', '它', '是', '软的'],
            "question":['葡萄', '是', '硬的', '吗'],
            "answer":1,
            "qa_id":5},
        {
            "passage": ['香蕉', '是', '甜的', '它', '是', '软的', '也是', '好吃的'],
            "question":['香蕉', '是', '好吃的', '吗'],
            "answer":0,
            "qa_id":6}
    ]

    train_2_examples = [
        {
            "passage": ['苹果'],
            "question": ['苹果'],
            "answer":0,
            "qa_id":7},
        {
            "passage": ['梨'],
            "question": ['西瓜'],
            "answer":1,
            "qa_id":8},
        {
            "passage": ['葡萄', '香蕉'],
            "question": ['葡萄', '香蕉'],
            "answer":0,
            "qa_id":9},
        {
            "passage": ['西瓜', '橘子'],
            "question": ['甜的'],
            "answer":1,
            "qa_id":10},
    ]

    dev_2_examples = [
        {
            "passage": ['橘子'],
            "question": ['苹果'],
            "answer":1,
            "qa_id":11},
        {
            "passage": ['梨', '西瓜'],
            "question": ['梨', '西瓜'],
            "answer":0,
            "qa_id":12},
    ]

    word2idx_dict = {"null":0,"苹果":1,"梨":2,"西瓜":3,"葡萄":4,"香蕉":5,"橘子":6,"甜的":7,"酸的":8,"硬的":9,\
        "软的":10,"大的":11,"圆的":12,"好吃的":13,"是":14,"也是":15,"它":16,"和":17,"吗":18}
    """
    id2vec = {
    0:[0.0,0.0,0.0,0.0],
    1:[0.1,0.1,0.1,0.1],
    2:[0.1,0.2,0.1,0.2],
    3:[0.2,0.1,0.3,0.2],
    4:[0.4,0.2,0.3,0.4],
    5:[0.4,0.4,0.4,0.4],
    6:[0.5,0.4,0.5,0.5],
    7:[0.6,0.5,0.5,0.6],
    8:[0.5,0.7,0.6,0.5],
    9:[0.7,0.6,0.5,0.5],
    10:[0.8,0.5,0.7,0.6],
    11:[0.8,0.6,0.6,0.6],
    12:[0.6,0.8,0.8,0.5],
    13:[0.5,0.8,0.8,0.6],
    14:[0.9,0.9,0.9,0.9],
    15:[0.9,0.9,0.8,0.8],
    17:[0.9,0.8,0.7,0.6],
    18:[0.9,0.5,0.6,0.7]
    }
    """

    id2vec = [
    [0.0,0.0,0.0,0.0],
    [0.05,0.05,0.05,0.05],
    [0.1,0.1,0.1,0.1],
    [0.15,0.15,0.15,0.15],
    [0.2,0.2,0.2,0.2],
    [0.25,0.25,0.25,0.25],
    [0.3,0.3,0.3,0.3],
    [0.35,0.35,0.35,0.35],
    [0.4,0.4,0.4,0.4],
    [0.45,0.46,0.45,0.45],
    [0.5,0.5,0.5,0.5],
    [0.55,0.55,0.55,0.55],
    [0.6,0.6,0.6,0.6],
    [0.65,0.65,0.65,0.65],
    [0.7,0.7,0.7,0.7],
    [0.75,0.75,0.75,0.75],
    [0.8,0.8,0.8,0.8],
    [0.85,0.85,0.85,0.85]
    ]

    print("Building model...")
    word_mat = np.array(id2vec)
    model = DemoModel(word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.learning_rate,
                           tf.constant(learning_rate, dtype=tf.float32)))

        dev_p, dev_q, dev_a, dev_id = get_bacth(dev_2_examples, word2idx_dict, batch_size)
        
        def get_acc(outputs, targets):
            t = 0
            for i in range(len(outputs)):
                if outputs[i] == targets[i]:
                    t += 1
            return (t / len(outputs)) * 1.0

        for i in range(10000):
            global_step = sess.run(model.global_step) + 1
            random.shuffle(train_examples)
            train_p, train_q, train_a, train_id = get_bacth(train_2_examples, word2idx_dict, batch_size)
            # train
            feed = {model.passage: train_p, model.question: train_q, model.answer: train_a, model.qa_id: train_id, model.is_train:True}
            train_loss, train_op, t_classes, t_id = sess.run([model.loss, model.train_op, model.classes, model.qa_id], feed_dict=feed)
            # dev
            feed2 = {model.passage: dev_p, model.question: dev_q, model.answer: dev_a, model.qa_id: dev_id, model.is_train:False}
            dev_loss, d_classes, d_id = sess.run([model.loss, model.classes, model.qa_id], feed_dict=feed2)
            # 输出
            train_acc = get_acc(t_classes, train_a)
            dev_acc = get_acc(d_classes, dev_a)
            if (i + 1) % 100 == 0:
                print("steps:{},train_loss:{:.4f},train_acc:{:.4f},dev_loss:{:.4f},dev_acc:{:.4f}"\
                    .format(global_step, train_loss, train_acc, dev_loss, dev_acc))
                for j in range(2):
                    print("dev_id:{},answer:{},my_answer:{}".format(dev_id[j],dev_a[j],d_classes[j]))


if __name__ == '__main__':
    tf.app.run()

