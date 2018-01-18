# coding=utf-8
# from tensorflow.models.rnn.translate import seq2seq_model
import numpy as np
import tensorflow as tf  # 0.12
from tqdm import tqdm

import utils.config as config
import utils.cornell_data_utils as cdu
from utils.model_utils import Chatbot


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
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



model_dir = 'C:/Users/Administrator/PycharmProjects/test/chatBot/checkpoint/'
ckpt = tf.train.get_checkpoint_state(model_dir)
if ckpt != None:
    f = open('word_to_id41.txt', 'r', encoding='utf-8')
    a = f.read()
    word_to_id = eval(a)

    f1 = open('id_to_word41.txt', 'r', encoding='utf-8')
    a1 = f1.read()
    id_to_word = eval(a1)
    vocab = list(word_to_id.keys())
else:
    vocab, word_to_id, id_to_word = cdu.create_vocab(cleaned_questions, cleaned_answers)
    file_object = open('word_to_id51.txt', 'w', encoding="utf8")
    file_object.write(str(word_to_id))
    file_object.close()

    file_object = open('id_to_word51.txt', 'w', encoding="utf8")
    file_object.write(str(id_to_word))
    file_object.close()



encoded_questions = cdu.encoder(cleaned_questions, word_to_id)
encoded_answers = cdu.encoder(cleaned_answers, word_to_id, True)

bucketed_data = cdu.bucket_data(encoded_questions, encoded_answers, word_to_id)

print('bucket prepared..')

####
# init
# model
model = Chatbot(config.LEARNING_RATE,
                config.BATCH_SIZE,
                config.ENCODING_EMBED_SIZE,
                config.DECODING_EMBED_SIZE,
                config.RNN_SIZE,
                config.NUM_LAYERS,
                len(vocab),
                word_to_id,
                id_to_word,
                config.CLIP_RATE,
                config.BUCKETS,
                )  # 4=clip_rate

############################################
####
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)
FLAGS = tf.app.flags.FLAGS


# 恢复前一次训练
if ckpt != None:
    print(ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
else:
    session.run(tf.global_variables_initializer())

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

            feed_dict = {model.inputs: X_batch,
                         model.targets: y_batch,
                         model.keep_probs: config.KEEP_PROBS,
                         model.decoder_seq_len: [len(y_batch[0])] * config.BATCH_SIZE,
                         model.encoder_seq_len: [len(X_batch[0])] * config.BATCH_SIZE}

            feed_dict2 = {model.inputs: X_batch,
                          model.keep_probs: config.KEEP_PROBS,
                          model.decoder_seq_len: [len(y_batch[0])] * config.BATCH_SIZE,
                          model.encoder_seq_len: [len(X_batch[0])] * config.BATCH_SIZE}
            cost, _, preds = session.run([model.loss, model.opt, model.predictions], feed_dict=feed_dict)

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
