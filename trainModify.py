# coding=utf-8
import tensorflow as tf  # 0.12
# from tensorflow.models.rnn.translate import seq2seq_model


import os
import numpy as np
import math
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
# 导入文件
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vec = 'train_encode.vec'
train_decode_vec = 'train_decode.vec'
test_encode_vec = 'test_encode.vec'
test_decode_vec = 'test_decode.vec'

# 词汇表大小5000
vocabulary_encode_size = 5000
vocabulary_decode_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3  # 层数
batch_size = 64





# str = unicode(str, errors='ignore')
# 读取*dencode.vec和*decode.vec数据（数据还不算太多, 一次读人到内存）
def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in buckets]  # 生成了[[],[],[],[]],即当值与参数不一样
    with tf.gfile.GFile(source_path, mode="r") as source_file:  # 以读格式打开源文件（source_file）
        with tf.gfile.GFile(target_path, mode="r") as target_file:  # 以读格式打开目标文件
            source, target = source_file.readline().encode('utf-8'), target_file.readline().encode('utf-8')  # 只读取一行
            counter = 0  # 计数器为0
            while source and target and (not max_size or counter < max_size):  # 当读入的还存在时
                counter += 1
                source_ids = [int(x) for x in source.split()]  # source的目标序列号，默认分隔符为空格，组成了一个源序列
                target_ids = [int(x) for x in target.split()]  # target组成一个目标序列，为目标序列
                target_ids.append(EOS_ID)  # 加上结束标记的序列号
                # for bucket_id, (source_size, target_size) in enumerate(buckets):  # enumerate()遍历序列中的元素和其下标
                #     if len(source_ids) < source_size and len(target_ids) < target_size:  # 判断是否超越了最大长度
                data_set.append([source_ids, target_ids])  # 读取到数据集文件中区
                # break  # 一次即可，跳出当前循环
                source, target = source_file.readline(), target_file.readline()  # 读取了下一行
    return data_set

#
# model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size, target_vocab_size=vocabulary_decode_size,
#                                    buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
#                                    batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97,
#                                    forward_only=False)

config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory

with tf.Session(config=config) as sess:
    # 恢复前一次训练
    # ckpt = tf.train.get_checkpoint_state('.')
    # if ckpt != None:
    #     print(ckpt.model_checkpoint_path)
    #     model.saver.restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     sess.run(tf.global_variables_initializer())

    train_set = read_data(train_encode_vec, train_decode_vec)
    test_set = read_data(test_encode_vec, test_decode_vec)
    train_set = train_set[4:]
    test_set = test_set[4:]


    train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]  # 分别计算出训练集中的长度【1,2,3,4】
    train_total_size = float(sum(train_bucket_sizes))  # 训练实例总数
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in
                           range(len(train_bucket_sizes))]  # 计算了之前所有的数的首战百分比

    loss = 0.0  # 损失置位0
    total_step = 0
    previous_losses = []
