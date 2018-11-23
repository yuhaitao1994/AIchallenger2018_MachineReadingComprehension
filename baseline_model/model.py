"""
AI Challenger观点型问题阅读理解

model.py：基于R-Net的改进模型，将PtrNet改成分类器

@author: yuhaitao
"""
# -*- coding:utf-8 -*-

import tensorflow as tf
from nn_func import cudnn_gru, native_gru, dot_attention, summ, dropout


class Model(object):
    def __init__(self, config, batch, word_mat=None, trainable=True, opt=True):
        """
        模型初始化函数
        Args:
            config:是tf.flag.FLAGS，配置整个项目的超参数
            batch:是一个tf.data.iterator对象，读取数据的迭代器，可能联系到tf.records，如果我们的数据集比较小就可以不用
            word_mat:np.array数组，是词向量？
            char_mat:同上
        """
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # tf.data.iterator的get_next方法，返回dataset中下一个element的tensor对象，在sess.run中实现迭代
        """
        passage: passage序列的每个词的id号的tensor(tf.int32)，长度应该是都取最大限制长度，空余的填充空值？(这里待定)
        question: question序列的每个词的id号的tensor(tf.int32)
        ch, qh, y1, y2: 本项目不需要，已经取消
        qa_id: question的id
        answer: 新添加的answer标签，(0/1/2)，shape初步定义为[batch_size]
        """
        self.passage, self.question, self.answer, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)

        # word embeddings的变量,这里定义的是不能训练的
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)

        # tf.cast将tensor转换为bool类型，生成mask，有值部分用true，空值用false
        self.c_mask = tf.cast(self.passage, tf.bool)
        self.q_mask = tf.cast(self.question, tf.bool)
        # 求每个序列的真实长度，得到_len的tensor
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            batch_size = config.batch_size
            # 求一个batch中序列最大长度，并按照最大长度对对tensor进行slice划分
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.passage, [0, 0], [
                              batch_size, self.c_maxlen])
            self.q = tf.slice(self.question, [0, 0], [
                              batch_size, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [
                                   batch_size, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [
                                   batch_size, self.q_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.RNet()  # 构造R-Net模型

        if trainable:

            self.learning_rate = tf.get_variable(
                "learning_rate", shape=[], dtype=tf.float32, trainable=False)

            if config.optimizer == "Adam":
                self.opt = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, epsilon=1e-8)
            elif config.optimizer == "Adadelta":
                self.opt = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate, epsilon=1e-6)
            else:
                print("optimizer error")
                return

            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

            # # 对embedding层设置单独的学习率
            # self.emb_lr = tf.get_variable(
            #     "emb_lr", shape=[], dtype=tf.float32, trainable=False)
            # self.learning_rate = tf.get_variable(
            #     "learning_rate", shape=[], dtype=tf.float32, trainable=False)
            # self.emb_opt = tf.train.AdamOptimizer(
            #     learning_rate=self.emb_lr, epsilon=1e-8)
            # self.opt = tf.train.AdamOptimizer(
            #     learning_rate=self.learning_rate, epsilon=1e-8)
            # # 区分不同的变量列表
            # self.var_list = tf.trainable_variables()
            # var_list1 = []
            # var_list2 = []
            # for var in self.var_list:
            #     if var.op.name == "word_mat":
            #         var_list1.append(var)
            #     else:
            #         var_list2.append(var)

            # grads = tf.gradients(self.loss, var_list1 + var_list2)
            # capped_grads, _ = tf.clip_by_global_norm(
            #     grads, config.grad_clip)
            # grads1 = capped_grads[:len(var_list1)]
            # grads2 = capped_grads[len(var_list1):]
            # self.train_op1 = self.emb_opt.apply_gradients(
            #     zip(grads1, var_list1))
            # self.train_op2 = self.opt.apply_gradients(
            #     zip(grads2, var_list2), global_step=self.global_step)
            # self.train_op = tf.group(self.train_op1, self.train_op2)

    def RNet(self):
        config = self.config
        batch_size, PL, QL, d = config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden
        gru = cudnn_gru if config.use_cudnn else native_gru  # 选择使用哪种gru网络

        with tf.variable_scope("embedding"):
            # word_embedding层
            with tf.name_scope("word"):
                # embedding后的shape是[batch_size, max_len, vec_len]
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

        with tf.variable_scope("encoding"):
            # encoder层，将context和question分别输入双向GRU
            rnn = gru(num_layers=3, num_units=d, batch_size=batch_size, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # RNN每层的正向反向输出合并，本代码默认的是每层的输出也合并
            # 所以对于3层rnn，输出的shape是[batch_size, max_len, 6*num_units]
            # 并且，序列空值处的输出都清零了
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("QP_attention"):
            """
            基于注意力的循环神经网络层，匹配context和question
            """
            # qc_att的shape [batch_size, c_maxlen, 12*hidden]
            qc_att_ = dot_attention(inputs=c, memory=q, mask=self.q_mask, hidden=d,
                                    keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=qc_att_.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # att:[batch_size, c_maxlen, 6*hidden]
            qc_att = rnn(qc_att_, seq_len=self.c_len)

        with tf.variable_scope("passage_match"):
            """
            context自匹配层
            """
            c_att = dot_attention(
                qc_att, qc_att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=batch_size, input_size=c_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # match:[batch_size, c_maxlen, 6*hidden]
            c_match = rnn(c_att, seq_len=self.c_len)

        with tf.variable_scope("YesNo_classification"):
            """
            对问题答案的分类层, 需要的输入有question的编码结果q和context的match
            """
            # init的shape:[batch_size, 2*hidden]
            # 这步的作用初始猜测是将question进行pooling操作，然后再输入给一个rnn层进行分类
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.keep_prob, is_train=self.is_train)
            c_match_ = dropout(c_match, keep_prob=config.keep_prob,
                               is_train=self.is_train)
            final_hiddens = init.get_shape().as_list()[-1]
            final_gru = tf.contrib.rnn.GRUCell(final_hiddens)
            _, final_state = tf.nn.dynamic_rnn(
                final_gru, c_match_, initial_state=init, dtype=tf.float32)
            final_w = tf.get_variable(name="final_w", shape=[final_hiddens, 3])
            final_b = tf.get_variable(name="final_b", shape=[
                                      3], initializer=tf.constant_initializer(0.))
            self.logits = tf.matmul(final_state, final_w)
            self.logits = tf.nn.bias_add(
                self.logits, final_b)  # logits:[batch_size, 3]

        with tf.variable_scope("softmax_and_loss"):
            final_softmax = tf.nn.softmax(self.logits)
            self.classes = tf.cast(
                tf.argmax(final_softmax, axis=1), dtype=tf.int32, name="classes")
            # 注意stop_gradient的使用，因为answer不是placeholder传进来的，所以要注明不对其计算梯度
            if config.loss_function == "focal_loss":
                self.loss = tf.reduce_mean(sparse_focal_loss(
                    logits=self.logits, labels=tf.stop_gradient(self.answer)))
            else:
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=tf.stop_gradient(self.answer)))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
