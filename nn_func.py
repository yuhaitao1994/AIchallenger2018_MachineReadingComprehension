"""
AI Challenger观点型问题阅读理解

nn_func.py：神经网络模型的组件

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import tensorflow as tf

INF = 1e30


class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        # cudnn GRU需要交换张量的维度，可能是便于计算
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            # 双向Bi-GRU f:forward b:back
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            # tf.tile 平铺给定的张量，这里是将初始状态扩张到batch_size倍
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, concat_layers=True):
        """
        运行RNN
        这里的keep_prob和is_train没用，在__init__中就已设置好了
        """
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                # 正向RNN
                with tf.variable_scope("fw_{}".format(layer)):
                    # 每一层使用上层的输出
                    # dynamic_rnn中的超过seq_len的部分就不计算了，state直接重复，output直接清零，节省资源
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                # 反向RNN
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                # 正向输出和反向输出合并
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


def dropout(args, keep_prob, is_train, mode="recurrent"):
    """
    dropout层,args初始是1.0
    """
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    """
    作用是给空值处减小注意力
    """
    return -INF * (1 - tf.cast(mask, tf.float32)) + val  # tf.cast:true转为1.0，false转为0.0


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    """
    对question进行最后一步的处理，可以看作是pooling吗
    """
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        # tf.squeeze把长度只有1的维度去掉
        # s1:[batch_size, c_maxlen]
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)  # 逐元素相乘，shape跟随memory一致
        return res  # [batch_size, 2*hidden]


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    """
    门控attention层
    """
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]  # inputs的1维度，应该是c_maxlen

        with tf.variable_scope("attention"):
            # inputs_的shape:[batch_size, c_maxlen, hidden]
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            # 三维矩阵相乘，结果的shape是[batch_size, c_maxlen, q_maxlen]
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            # 将mask平铺成与outputs相同的形状，这里考虑，改进成input和memory都需要mask
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            outputs = tf.matmul(logits, memory)
            # res:[batch_size, c_maxlen, 12*hidden]
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            """
            attention * gate
            """
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate  # 向量的逐元素相乘


# 写一个谷歌论文中新的attention模块
def multihead_attention(Q, K, V, mask, hidden, head_num=4, keep_prob=1.0, is_train=None, has_gate=True, scope="multihead_attention"):
    """
    Q : passage
    K,V: question
    mask: Q的mask
    """
    size = int(hidden / head_num)  # 每个attention的大小

    with tf.variable_scope(scope):
        d_Q = dropout(Q, keep_prob=keep_prob, is_train=is_train)
        d_K = dropout(K, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(Q)[1]

        with tf.variable_scope("attention"):
            Q_ = tf.nn.relu(dense(d_Q, hidden, use_bias=False, scope="Q"))
            K_ = tf.nn.relu(dense(d_K, hidden, use_bias=False, scope="K"))
            V_ = tf.nn.relu(dense(V, hidden, use_bias=False, scope="V"))
            Q_ = tf.reshape(Q_, (-1, tf.shape(Q_)[1], head_num, size))
            K_ = tf.reshape(K_, (-1, tf.shape(K_)[1], head_num, size))
            V_ = tf.reshape(V_, (-1, tf.shape(V_)[1], head_num, size))
            Q_ = tf.transpose(Q_, [0, 2, 1, 3])
            K_ = tf.transpose(K_, [0, 2, 1, 3])
            V_ = tf.transpose(V_, [0, 2, 1, 3])
            # scale:[batch_size, head_num, c_maxlen, q_maxlen]
            scale = tf.matmul(Q_, K_, transpose_b=True) / tf.sqrt(float(size))
            scale = tf.transpose(scale, [0, 3, 2, 1])
            for _ in range(len(scale.shape) - 2):
                mask = tf.expand_dims(mask, axis=2)
            mask_scale = softmax_mask(scale, mask)
            mask_scale = tf.transpose(scale, [0, 3, 2, 1])
            logits = tf.nn.softmax(mask_scale)
            outputs = tf.matmul(logits, V_)  # [b,h,c,s]
            outputs = tf.transpose(outputs, [0, 2, 1, 3])
            # [batch_size, c_maxlen, hidden]
            outputs = tf.reshape(outputs, (-1, tf.shape(Q)[1], hidden))
            # res连接
            res = tf.concat([Q, outputs], axis=2)

        if has_gate:
            with tf.variable_scope("gate"):
                dim = res.get_shape().as_list()[-1]
                d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
                gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
                return res * gate
        else:
            return res


def dense(inputs, hidden, use_bias=True, scope="dense"):
    """
    全连接层
    """
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        # 三维的inputs，reshape成二维
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        # outshape就是input的最后一维变成hidden
        res = tf.reshape(res, out_shape)
        return res
