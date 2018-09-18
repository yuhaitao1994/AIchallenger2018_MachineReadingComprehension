import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string


def get_record_parser(config):
    def parse(example):
        para_limit = config.para_limit
        ques_limit = config.ques_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "passage_idxs": tf.FixedLenFeature([], tf.string),
                                               "question_idxs": tf.FixedLenFeature([], tf.string),
                                               "answer": tf.FixedLenFeature([], tf.int64),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        # tf.decode_raw: 将字符串的字节重新解释为数字向量
        passage_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        question_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        answer = features["answer"]
        qa_id = features["id"]
        return passage_idxs, question_idxs, answer, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    """
    训练数据集TFRecordDataset的batch生成器。
    Args:
        record_file: 训练数据tf_record路径
        parser: 数据存储的格式
        config: 超参数
    """
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        # bucket方法，用于解决序列长度不同的mini-batch的计算效率问题
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


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
