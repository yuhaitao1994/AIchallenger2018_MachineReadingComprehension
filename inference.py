"""
AI Challenger观点型问题阅读理解

inference.py：模型测试代码

@author: yuhaitao
"""
# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np

from func import cudnn_gru, native_gru, dot_attention, summ, ptr_net

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow的log显示级别
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Must be consistant with training
hidden = 75
use_cudnn = False
batch_size = 4


class InfModel(object):
    # Used to zero elements in the probability matrix that correspond to answer
    # spans that are longer than the number of tokens specified here.

    def __init__(self, word_mat, char_mat, trainable=True, opt=True):
        self.c = tf.placeholder(tf.int32, [batch_size, None])
        self.q = tf.placeholder(tf.int32, [batch_size, None])
        self.answer = tf.placeholder(tf.int32, [batch_size])
        self.qa_id = tf.placeholder(tf.int32, [batch_size])
        self.is_train = tf.placeholder(
            tf.bool, shape=[], dtype=tf.bool, trainable=False)
        self.tokens_in_context = tf.placeholder(tf.int64)

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [batch_size, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [batch_size, self.q_maxlen])
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
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def RNet(self):
        PL, QL, d = self.c_maxlen, self.q_maxlenmit, hidden
        gru = cudnn_gru if use_cudnn else native_gru

        with tf.variable_scope("embedding"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=batch_size, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(inputs=c, memory=q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # match:[batch_size, c_maxlen, 6*hidden]
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("YesNo_classification"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            match = dropout(match, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            final_hiddens = init.get_shape().as_list()[-1]
            final_gru = tf.contrib.rnn.GRUCell(final_hiddens)
            _, final_state = final_gru(match, init)
            final_w = tf.get_variable(name="final_w", shape=[final_hiddens, 3])
            final_b = tf.get_variable(name="final_b", shape=[
                                      3], initializer=tf.constant_initializer(0.))
            logits = tf.matmul(final_state, final_w)
            logits = tf.nn.bias_add(logits, final_b)  # logits:[batch_size, 3]

        with tf.variable_scope("softmax_and_loss"):
            final_softmax = tf.nn.softmax(logits)
            self.classes = tf.cast(
                tf.argmax(logits, axis=1), dtype=tf.int32, name="classes")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=tf.stop_gradient(self.answer)))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step


if __name__ == '__main__':
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

    word2idx_dict = {"null":0,"苹果":1,"梨":2,"西瓜":3,"葡萄":4,"香蕉":5,"橘子":6,"甜的":7,"酸的":8,"硬的":9,\
        "软的":10,"大的":11,"圆的":12,"好吃的":13,"是":14,"也是":15,"它":16,"和":17,"吗":18}

    id2vec = {
    0:[0.0,0.0,0.0,0.0],
    1:[],
    
    }


class Inference(object):

    def __init__(self):
        with open(word_emb_file, "r") as fh:
            self.word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(char_emb_file, "r") as fh:
            self.char_mat = np.array(json.load(fh), dtype=np.float32)
        with open(word2idx_file, "r") as fh:
            self.word2idx_dict = json.load(fh)
        with open(char2idx_file, "r") as fh:
            self.char2idx_dict = json.load(fh)
        self.model = InfModel(self.word_mat, self.char_mat)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

    def response(self, context, question):
        sess = self.sess
        model = self.model
        span, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs = \
            self.prepro(context, question)
        yp1, yp2 = \
            sess.run(
                [model.yp1, model.yp2],
                feed_dict={
                    model.c: context_idxs, model.q: ques_idxs,
                    model.ch: context_char_idxs, model.qh: ques_char_idxs,
                    model.tokens_in_context: len(span)})
        start_idx = span[yp1[0]][0]
        end_idx = span[yp2[0]][1]
        return context[start_idx: end_idx]

    def prepro(self, context, question):
        context = context.replace("''", '" ').replace("``", '" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        ques = question.replace("''", '" ').replace("``", '" ')
        ques_tokens = word_tokenize(ques)
        ques_chars = [list(token) for token in ques_tokens]

        context_idxs = np.zeros([1, len(context_tokens)], dtype=np.int32)
        context_char_idxs = np.zeros(
            [1, len(context_tokens), char_limit], dtype=np.int32)
        ques_idxs = np.zeros([1, len(ques_tokens)], dtype=np.int32)
        ques_char_idxs = np.zeros(
            [1, len(ques_tokens), char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word2idx_dict:
                    return self.word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in self.char2idx_dict:
                return self.char2idx_dict[char]
            return 1

        for i, token in enumerate(context_tokens):
            context_idxs[0, i] = _get_word(token)

        for i, token in enumerate(ques_tokens):
            ques_idxs[0, i] = _get_word(token)

        for i, token in enumerate(context_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[0, i, j] = _get_char(char)

        for i, token in enumerate(ques_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[0, i, j] = _get_char(char)
        return spans, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs


# Demo, example from paper "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
if __name__ == "__main__":
    infer = Inference()
    context = "In meteorology, precipitation is any product of the condensation " \
              "of atmospheric water vapor that falls under gravity. The main forms " \
              "of precipitation include drizzle, rain, sleet, snow, graupel and hail." \
              "Precipitation forms as smaller droplets coalesce via collision with other " \
              "rain drops or ice crystals within a cloud. Short, intense periods of rain " \
              "in scattered locations are called “showers”."
    ques1 = "What causes precipitation to fall?"
    ques2 = "What is another main form of precipitation besides drizzle, rain, snow, sleet and hail?"
    ques3 = "Where do water droplets collide with ice crystals to form precipitation?"

    # Correct: gravity, Output: drizzle, rain, sleet, snow, graupel and hail
    ans1 = infer.response(context, ques1)
    print("Answer 1: {}".format(ans1))

    # Correct: graupel, Output: graupel
    ans2 = infer.response(context, ques2)
    print("Answer 2: {}".format(ans2))

    # Correct: within a cloud, Output: within a cloud
    ans3 = infer.response(context, ques3)
    print("Answer 3: {}".format(ans3))
