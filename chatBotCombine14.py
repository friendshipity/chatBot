# coding=utf-8
import tensorflow as tf  # 0.12
# from tensorflow.models.rnn.translate import seq2seq_model
import numpy as np
import utils.helpers
import utils.model_utils as mu
import utils.cornell_data_utils as cdu
from tqdm import tqdm
import utils.config as config

logs_path = 'C:/Users/Administrator/PycharmProjects/test/chatBot/trainTest/'
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
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

src_vocab_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3  # 层数
batch_size = 64

VOCAB_THRESHOLD = 5000
BUCKETS = [(10, 15), (15, 25), (25, 45), (45, 60), (60, 100)]  # First try buckets you can tweak these
EPOCHS = 100
BATCH_SIZE = 64
RNN_SIZE = 256  # rnn layer input size
NUM_LAYERS = 3  # rnn layer size
ENCODING_EMBED_SIZE = 512
DECODING_EMBED_SIZE = 512
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9  # nisam siguran da cu ovo koristiti
MIN_LEARNING_RATE = 0.0001
KEEP_PROBS = 0.5  # dropout rate
CLIP_RATE = 4

####################
# word map construct
# train_enc_word_to_id,train_enc_id_to_word = mu.vocab_dict('train_encode_vocabulary')
# train_dec_word_to_id,train_dec_id_to_word = mu.vocab_dict('train_decode_vocabulary')
train_encode_file = 'test.enc'
train_decode_file = 'test.dec'
cleaned_questions = []  # 对话集合
cleaned_answers = []  # 对话集合
with open(train_encode_file, encoding="utf8") as f:
    for line in f:
        cleaned_questions.append(line)

with open(train_decode_file, encoding="utf8") as f:
    for line in f:
        cleaned_answers.append(line)

vocab, word_to_id, id_to_word = cdu.create_vocab(cleaned_questions, cleaned_answers)

# read data
# def read_data(source_path, target_path, max_size=None):
#     data_set = [[] for _ in buckets]  # 生成了[[],[],[],[]],即当值与参数不一样
#     with tf.gfile.GFile(source_path, mode="r") as source_file:  # 以读格式打开源文件（source_file）
#         with tf.gfile.GFile(target_path, mode="r") as target_file:  # 以读格式打开目标文件
#             source, target = source_file.readline().encode('utf-8'), target_file.readline().encode('utf-8')  # 只读取一行
#             counter = 0  # 计数器为0
#             while source and target and (not max_size or counter < max_size):  # 当读入的还存在时
#                 counter += 1
#                 source_ids = [int(x) for x in source.split()]  # source的目标序列号，默认分隔符为空格，组成了一个源序列
#                 target_ids = [int(x) for x in target.split()]  # target组成一个目标序列，为目标序列
#                 target_ids.append(EOS_ID)  # 加上结束标记的序列号
#                 data_set.append([source_ids, target_ids])  # 读取到数据集文件中区
#                 source, target = source_file.readline(), target_file.readline()  # 读取了下一行
#     return data_set


# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory

# with tf.Session(config=config) as sess:
#     # 恢复前一次训练
#     # ckpt = tf.train.get_checkpoint_state('.')
#     # if ckpt != None:
#     #     print(ckpt.model_checkpoint_path)
#     #     model.saver.restore(sess, ckpt.model_checkpoint_path)
#     # else:
#     #     sess.run(tf.global_variables_initializer())
#
#     train_set = read_data(train_encode_vec, train_decode_vec)
#     test_set = read_data(test_encode_vec, test_decode_vec)
#     train_set = train_set[4:]
#     test_set = test_set[4:]



# encoded_questions = open('train_encode.vec','r',encoding='utf-8')
# encoded_answers = open('train_decode.vec','r',encoding='utf-8')
encoded_questions = cdu.encoder(cleaned_questions, word_to_id)
encoded_answers = cdu.encoder(cleaned_answers, word_to_id, True)

bucketed_data = cdu.bucket_data(encoded_questions, encoded_answers, word_to_id)

print('bucket prepared..')

# model
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

####
# init
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
keep_probs = tf.placeholder(tf.float32, name='dropout_rate')
encoder_seq_len = tf.placeholder(tf.int32, (None,), name='encoder_seq_len')
decoder_seq_len = tf.placeholder(tf.int32, (None,), name='decoder_seq_len')
max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')
####
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

#    model

###############################################################################
#    encoder


enc_outputs, enc_states = mu.encoder(encoder_inputs,
                                     RNN_SIZE,
                                     NUM_LAYERS,
                                     encoder_seq_len,
                                     keep_probs,
                                     ENCODING_EMBED_SIZE,
                                     src_vocab_size)

###############################################################################
#    attention
dec_inputs = mu.decoder_inputs_preprocessing(decoder_targets,
                                             word_to_id,
                                             batch_size)

decoder_cell, encoder_states_new = mu.attention_mech(RNN_SIZE,
                                                     keep_probs,
                                                     enc_outputs,
                                                     enc_states,
                                                     encoder_seq_len,
                                                     batch_size)

###############################################################################
#    decoder

train_outputs, inference_output = mu.decoder(dec_inputs,
                                             encoder_states_new,
                                             decoder_cell,
                                             DECODING_EMBED_SIZE,
                                             src_vocab_size,
                                             decoder_seq_len,
                                             max_seq_len,
                                             word_to_id,
                                             batch_size)

###############################################################################
#   prediction loss opt

predictions = tf.identity(inference_output.sample_id, name='preds')

loss, opt = mu.opt_loss(train_outputs,
                        decoder_targets,
                        decoder_seq_len,
                        max_seq_len,
                        LEARNING_RATE,
                        CLIP_RATE)

############################################
####
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)
for i in range(config.EPOCHS):
    epoch_accuracy = []
    epoch_loss = []
    for b in range(len(bucketed_data)):
        bucket = bucketed_data[b]
        questions_bucket = []
        answers_bucket = []
        bucket_accuracy = []
        bucket_loss = []
        for k in range(len(bucket)):
            questions_bucket.append(np.array(bucket[k][0]))
            answers_bucket.append(np.array(bucket[k][1]))

        for ii in tqdm(range(len(questions_bucket) // config.BATCH_SIZE)):
            starting_id = ii * config.BATCH_SIZE

            X_batch = questions_bucket[starting_id:starting_id + config.BATCH_SIZE]
            y_batch = answers_bucket[starting_id:starting_id + config.BATCH_SIZE]

            feed_dict = {encoder_inputs: X_batch,
                         decoder_targets: y_batch,
                         keep_probs: config.KEEP_PROBS,
                         decoder_seq_len: [len(y_batch[0])] * config.BATCH_SIZE,
                         encoder_seq_len: [len(X_batch[0])] * config.BATCH_SIZE}



            cost, _, preds = session.run([loss, opt, predictions], feed_dict=feed_dict)

            epoch_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))
            bucket_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))

            bucket_loss.append(cost)
            epoch_loss.append(cost)

        print("Bucket {}:".format(b + 1),
              " | Loss: {}".format(np.mean(bucket_loss)),
              " | Accuracy: {}".format(np.mean(bucket_accuracy)))

    print("EPOCH: {}/{}".format(i, config.EPOCHS),
          " | Epoch loss: {}".format(np.mean(epoch_loss)),
          " | Epoch accuracy: {}".format(np.mean(epoch_accuracy)))

    saver.save(session, "checkpoint/chatbot_{}.ckpt".format(i))



#
#
#
# loss = tf.reduce_mean(stepwise_cross_entropy)
# train_op = tf.train.AdamOptimizer().minimize(loss)
#
# sess.run(tf.global_variables_initializer())
#
# batch_ = [[6], [3, 4], [9, 8, 7]]
#
# batch_, batch_length_ = helpers.batch(batch_)
# print('batch_encoded:\n' + str(batch_))
#
# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                             max_sequence_length=4)
# print('decoder inputs:\n' + str(din_))
#
# pred_ = sess.run(decoder_prediction,
#                  feed_dict={
#                      encoder_inputs: batch_,
#                      decoder_inputs: din_,
#                  })
# print('decoder predictions:\n' + str(pred_))
#
# ############## training on the toy task
#
#
# batch_size = 100
#
# batches = helpers.random_sequences(length_from=3, length_to=8,
#                                    vocab_lower=2, vocab_upper=10,
#                                    batch_size=batch_size)
#
# print('head of the batch:')
# for seq in next(batches)[:10]:
#     print(seq)
#
#
# def next_feed():
#     batch = next(batches)
#     encoder_inputs_, _ = helpers.batch(batch)
#     decoder_targets_, _ = helpers.batch(
#         [(sequence) + [EOS_ID] for sequence in batch]
#     )
#     decoder_inputs_, _ = helpers.batch(
#         [[EOS_ID] + (sequence) for sequence in batch]
#     )
#     return {
#         encoder_inputs: encoder_inputs_,
#         decoder_inputs: decoder_inputs_,
#         decoder_targets: decoder_targets_,
#     }
#
#
# loss_track = []
# max_batches = 3001
# batches_in_epoch = 1000
# summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#
# try:
#     for batch in range(max_batches):
#         fd = next_feed()
#         _, l = sess.run([train_op, loss], fd)
#         loss_track.append(l)
#
#         if batch == 0 or batch % batches_in_epoch == 0:
#             print('batch {}'.format(batch))
#             print('  minibatch loss: {}'.format(sess.run(loss, fd)))
#
#             # op to write logs to Tensorboard
#             predict_ = sess.run(decoder_prediction, fd)
#             for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
#                 print('  sample {}:'.format(i + 1))
#                 print('    input     > {}'.format(inp))
#                 print('    predicted > {}'.format(pred))
#                 if i >= 2:
#                     break
#             print()
# except KeyboardInterrupt:
#     print('training interrupted')
