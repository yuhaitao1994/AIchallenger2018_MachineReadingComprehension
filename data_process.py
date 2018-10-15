"""
AI Challenger观点型问题阅读理解

data_process.py：数据预处理代码

@author: haomaojie
"""
# -*- coding:utf-8 -*-
import pandas as pd
import time
import json
import jieba
import csv
import word2vec
import re
import tensorflow as tf
import numpy as np
from tqdm import tqdm  # 进度条
import os
import gensim


def read_data(json_path, output_path, line_count):
    '''
    读取json文件并转成Dataframe
    '''
    start_time = time.time()
    data = []
    with open(json_path, 'r') as f:
        for i in range(line_count):
            data_list = json.loads(f.readline())
            data.append([data_list['passage'], data_list['query']])
        df = pd.DataFrame(data, columns=['passage', 'query'])
    df.to_csv(output_path, index=False)
    print('转化成功，已生成csv文件')
    end_time = time.time()
    print(end_time - start_time)


def de_word(data_path, out_path):
    '''
    分词
    '''
    start_time = time.time()
    word = []
    data_file = open(data_path).read().split('\n')
    for i in range(len(data_file)):
        result = []
        seg_list = jieba.cut(data_file[i])
        for w in seg_list:
            result.append(w)
        word.append(result)
    print('分词完成')
    with open(out_path, 'w+') as txt_write:
        for i in range(len(word)):
            s = str(word[i]).replace(
                '[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + \
                '\n'  # 去除单引号，逗号，每行末尾追加换行符
            txt_write.write(s)
    print('保存成功')
    end_time = time.time()
    print(end_time - start_time)


def word_vec(file_txt, file_bin, min_count, size):
    word2vec.word2vec(file_txt, file_bin, min_count=min_count,
                      size=size, verbose=True)


def merge_csv(target_dir, output_file):
    for inputfile in [os.path.join(target_dir, 'train_oridata.csv'),
                      os.path.join(target_dir, 'test_oridata.csv'), os.path.join(target_dir, 'validation_oridata.csv')]:
        data = pd.read_csv(inputfile)
        df = pd.DataFrame(data)
        df.to_csv(output_file, mode='a', index=False)

# 词转id，id转向量


def transfer(model_path, embedding_size):
    start_time = time.time()
    model = word2vec.load(model_path)
    word2id_dic = {}
    init_0 = [0.0 for i in range(embedding_size)]
    id2vec_dic = [init_0]
    for i in range(len(model.vocab)):
        id = i + 1
        word2id_dic[model.vocab[i]] = id
        id2vec_dic.append(model[model.vocab[i]].tolist())
    end_time = time.time()
    print('词转id，id转向量完成')
    print(end_time - start_time)
    return word2id_dic, id2vec_dic


def transfer_txt(model_path, embedding_size):
    print("开始转换...")
    start_time = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(
        model_path, binary=False)
    word_dic = model.wv.vocab
    word2id_dic = {}
    init_0 = [0.0 for i in range(embedding_size)]
    id2vec_dic = [init_0]
    id = 1
    for i in word_dic:
        word2id_dic[i] = id
        id2vec_dic.append(model[i].tolist())
        id += 1
    end_time = time.time()
    print('词转id，id转向量完成')
    print(end_time - start_time)
    return word2id_dic, id2vec_dic

# 存入json文件


def save_json(output_path, dic_data, message=None):
    start_time = time.time()
    if message is not None:
        print("Saving {}...".format(message))
        with open(output_path, "w") as fh:
            json.dump(dic_data, fh, ensure_ascii=False, indent=4)
    print('保存完成')
    end_time = time.time()
    print(end_time - start_time)

# 将原文中的passage，query，alternative，answer，query_id转成id号
# 输入参数为词典的位置和训练集的位置


def TrainningsetProcess(dic_url, dataset_url, passage_len_limit):
    res = []  # 最后返回的结果
    rule = re.compile(r'\|')
    id2alternatives = {}
    # 读取字典
    with open(dic_url, 'r', encoding='utf-8') as dic_file:
        dic = dict()
        dic = json.load(dic_file)
    # 读取训练集
    over_limit = 0
    with open(dataset_url, 'r', encoding='utf-8') as ts_file:
        for file_line in ts_file:
            line = json.loads(file_line)  # 读取一行json文件
            this_line_res = dict()  # 变量定义，代表这一行映射之后的结果
            passage = line['passage']
            alternatives = line['alternatives']
            query = line['query']
            if dataset_url.find('test') == -1:
                answer = line['answer']
            query_idx = line['query_id']

            # 用jieba将passage和query分词,lcut返回list
            passage_cut = jieba.lcut(passage, cut_all=False)
            query_cut = jieba.lcut(query, cut_all=False)

            # 用词典将passage和query映射到id
            passage_id = []
            query_id = []
            for each_passage_word in passage_cut:
                passage_id.append(dic.get(each_passage_word))
            for each_query_word in query_cut:
                query_id.append(dic.get(each_query_word))

            # 对选项进行排序
            alternatives_cut = re.split(rule, alternatives)
            alternatives_cut = [s.strip() for s in alternatives_cut]
            tmp = [0, 0, 0]

            # 选项少于三个
            if len(alternatives_cut) == 1:
                alternatives_cut.append(alternatives_cut[0])
                alternatives_cut.append(alternatives_cut[0])
            if len(alternatives_cut) == 2:
                alternatives_cut.append(alternatives_cut[0])

            # 跳过无效数据（135条）
            if alternatives.find("无法") == -1 and alternatives.find("不确定") == -1:
                if dataset_url.find('test') != -1:
                    tmp[0] = alternatives_cut[0]
                    tmp[1] = alternatives_cut[1]
                    tmp[2] = alternatives_cut[2]
                else:
                    print(1)
                    continue
            if alternatives.count("无法确定") > 1 or alternatives.count("没") > 1:
                if dataset_url.find('test') != -1:
                    tmp[0] = alternatives_cut[0]
                    tmp[1] = alternatives_cut[1]
                    tmp[2] = alternatives_cut[2]
                else:
                    print(2)
                    continue  # 第64772条数据
            if alternatives.find("没") != -1 and alternatives.find("不") != -1 and alternatives.find("不确定") == -1:
                print(3)
                continue  # 第144146条数据
            if "不确定" in alternatives_cut and "无法确定" in alternatives_cut:
                tmp[0] = "确定"
                tmp[1] = "不确定"
                tmp[2] = "无法确定"
            # 肯定/否定/无法确定
            elif alternatives.find("不") != -1 or alternatives.find("没") != -1:
                if alternatives.count("不") == 1 and alternatives.find("不确定") != -1:
                    alternatives_cut.remove("不确定")
                    alternatives_cut.append("不确定")
                    tmp[0] = alternatives_cut[0]
                    tmp[1] = alternatives_cut[1]
                    tmp[2] = alternatives_cut[2]
                elif alternatives.count("不") > 1:
                    if alternatives.find("不确定") == -1:
                        if dataset_url.find("test") != -1:
                            tmp[0] = alternatives_cut[0]
                            tmp[1] = alternatives_cut[1]
                            tmp[2] = alternatives_cut[2]
                        else:
                            print(line)
                            continue
                    else:
                        alternatives_cut.remove("不确定")
                        if alternatives_cut[0].find("不") != -1:
                            tmp[1] = alternatives_cut[0]
                            tmp[0] = alternatives_cut[1]
                        else:
                            tmp[1] = alternatives_cut[1]
                            tmp[0] = alternatives_cut[0]
                        alternatives_cut.append("不确定")
                        tmp[2] = alternatives_cut[2]
                else:
                    for tmp_alternatives in alternatives_cut:
                        if tmp_alternatives.find("无法") != -1:
                            tmp[2] = tmp_alternatives
                        elif tmp_alternatives.find("不") != -1 or tmp_alternatives.find("没") != -1:
                            tmp[1] = tmp_alternatives
                        else:
                            tmp[0] = tmp_alternatives
            # 无明显肯定与否定词义
            else:
                for tmp_alternatives in alternatives_cut:
                    if tmp_alternatives.find("无法") != -1 or alternatives.find("不确定") != -1:
                        alternatives_cut.remove(tmp_alternatives)
                        alternatives_cut.append(tmp_alternatives)
                        break
                tmp[0] = alternatives_cut[0]
                tmp[1] = alternatives_cut[1]
                tmp[2] = alternatives_cut[2]

            # 根据tmp列表生成answer_id
            if dataset_url.find('test') == -1:
                answer_id = tmp.index(answer.strip())
            # 得到这一行映射后的结果，是dict类型的数据
            if len(passage_id) > passage_len_limit:
                passage_id = passage_id[:passage_len_limit]
                over_limit += 1
            this_line_res['passage'] = passage_id
            this_line_res['query'] = query_id
            this_line_res['alternatives'] = tmp
            if dataset_url.find('test') == -1:
                this_line_res['answer'] = answer_id
            this_line_res['query_id'] = query_idx
            # 创建query_id到alternatives的字典，保存为json
            id2alternatives[query_idx] = tmp
            res.append(this_line_res)
        print(len(res))
        print("over_limit:{}".format(over_limit))
        return res, id2alternatives


def data_process(config):
    target_dir = config.target_dir
    # 这里如果使用自己训练好的词向量就可以注释掉
    '''
    read_data(config.train_file, os.path.join(
        target_dir, 'train_oridata.csv'), 250000)  # 250000
    read_data(config.test_file, os.path.join(
        target_dir, 'test_oridata.csv'), 10000)  # 10000
    read_data(config.dev_file, os.path.join(
        target_dir, 'validation_oridata.csv'), 30000)  # 30000
    merge_csv(target_dir, os.path.join(target_dir, 'ori_data.csv'))
    de_word(os.path.join(target_dir, 'ori_data.csv'),
            os.path.join(target_dir, 'seg_list.txt'))
    word_vec(os.path.join(target_dir, 'seg_list.txt'),
             os.path.join(target_dir, 'seg_listWord2Vec.bin'), config.min_count, config.embedding_size)
    # 如果是用外部词向量，从这里开始
    # word2id_dic, id2vec_dic = transfer_txt(
    #     os.path.join(target_dir, 'baidu_300_wc+ng_sgns.baidubaike.bigram-char.txt'), config.embedding_size)
    word2id_dic, id2vec_dic = transfer(
        os.path.join(target_dir, 'seg_listWord2Vec.bin'), config.embedding_size)
    save_json(config.word2id_file, word2id_dic, "word to id")
    save_json(config.id2vec_file, id2vec_dic, "id to vec")
    '''
    train_examples, train_id2alternatives = TrainningsetProcess(
        config.word2id_file, config.train_file, config.para_limit)
    test_examples, test_id2alternatives = TrainningsetProcess(
        config.word2id_file, config.test_file, config.para_limit)
    validation_examples, validation_id2alternatives = TrainningsetProcess(
        config.word2id_file, config.dev_file, config.para_limit)
    save_json(config.train_eval_file, train_id2alternatives,
              message='保存train每条数据的alternatives')
    save_json(config.test_eval_file, test_id2alternatives,
              message='保存test每条数据的alternatives')
    save_json(config.dev_eval_file, validation_id2alternatives,
              message='保存validation每条数据的alternatives')
    return train_examples, test_examples, validation_examples


def build_features(config, examples, data_type, out_file, is_test=False):
    """
    将数据读入TFrecords
    """

    para_limit = config.para_limit
    ques_limit = config.ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    for example in tqdm(examples):
        total += 1
        passage_idxs = np.zeros([para_limit], dtype=np.int32)
        question_idxs = np.zeros([ques_limit], dtype=np.int32)

        for i, token in enumerate(example["passage"]):
            if token == None:
                passage_idxs[i] = 0
            else:
                passage_idxs[i] = token
        for i, token in enumerate(example["query"]):
            if token == None:
                question_idxs[i] = 0
            else:
                question_idxs[i] = token
        # print(passage_idxs)
        # print(example["passage"])
        if not is_test:
            record = tf.train.Example(features=tf.train.Features(feature={
                                      "passage_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_idxs.tostring()])),
                                      "question_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[question_idxs.tostring()])),
                                      "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["answer"]])),
                                      "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["query_id"]]))
                                      }))
        else:
            record = tf.train.Example(features=tf.train.Features(feature={
                                      "passage_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_idxs.tostring()])),
                                      "question_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[question_idxs.tostring()])),
                                      "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(-1)])),
                                      "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["query_id"]]))
                                      }))
        # print(record)
        writer.write(record.SerializeToString())
    print("Build {} instances of features in total".format(total))
    writer.close()


def prepro(config):
    """
    数据预处理函数
    """
    train_examples, test_examples, dev_examples = data_process(config)
    '''
    print(train_examples)
    print(test_examples)
    print(dev_examples)
    print(word2id_dict)
    '''
    # train: 249778, test: 10000, dev: 29968
    # train: 439, test: 18, dev: 48

    build_features(config, train_examples, "train", config.train_record_file)
    build_features(config, dev_examples, "dev", config.dev_record_file)
    build_features(config, test_examples, "test",
                   config.test_record_file, is_test=True)

    print("done!!!")
