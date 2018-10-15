"""
AI Challenger观点型问题阅读理解

config.py：配置文件，程序运行入口

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import os
import tensorflow as tf

import data_process
from main import train, test
from examine_dev import examine_dev

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

train_file = os.path.join("file", "ai_challenger_oqmrc_trainingset.json")
dev_file = os.path.join("file", "ai_challenger_oqmrc_validationset.json")
test_file = os.path.join("file", "ai_challenger_oqmrc_testa.json")
'''
train_file = os.path.join("file", "train_demo.json")
dev_file = os.path.join("file", "val_demo.json")
test_file = os.path.join("file", "test_demo.json")'''

target_dir = "data"
log_dir = "log/event"
save_dir = "log/model"
prediction_dir = "log/prediction"
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
id2vec_file = os.path.join(target_dir, "id2vec.json")  # id号->向量
word2id_file = os.path.join(target_dir, "word2id.json")  # 词->id号
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")
flags.DEFINE_string("gpu", "0", "0/1")
flags.DEFINE_string("experiment", "lalala", "每次存不同模型分不同的文件夹")
flags.DEFINE_string("model_name", "default", "选取不同的模型")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("prediction_dir", prediction_dir, "")
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")
flags.DEFINE_string("train_eval_file", train_eval, "")
flags.DEFINE_string("dev_eval_file", dev_eval, "")
flags.DEFINE_string("test_eval_file", test_eval, "")
flags.DEFINE_string("word2id_file", word2id_file, "")
flags.DEFINE_string("id2vec_file", id2vec_file, "")

flags.DEFINE_integer("para_limit", 150, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 30, "Limit length for question")
flags.DEFINE_integer("min_count", 1, "embedding 的最小出现次数")
flags.DEFINE_integer("embedding_size", 300, "the dimension of vector")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
# 使用cudnn训练，提升6倍速度
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn (only for GPU)")
flags.DEFINE_boolean("is_bucket", False, "Whether to use bucketing")
flags.DEFINE_list("bucket_range", [40, 361, 40], "range of bucket")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_steps", 200000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint for evaluation")
flags.DEFINE_integer("period", 500, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Num of batches for evaluation")
flags.DEFINE_float("init_learning_rate", 0.001,
                   "Initial learning rate for Adam")
flags.DEFINE_float("keep_prob", 0.7, "Keep prob in rnn")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 60, "Hidden size")  # best:128
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")
flags.DEFINE_string("optimizer", "Adam", "")
flags.DEFINE_string("loss_function", "default", "")


def main(_):
    config = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # 选择一块gpu
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        data_process.prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "examine":
        examine_dev(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
