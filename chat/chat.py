import os
import sys

import tensorflow as tf

from utils.config import FLAGS
from chat import data_utils
from chat.seq2seq_model_utils import create_model, get_predicted_sentence_1,get_predicted_sentence_2
import utils.cornell_data_utils as cdu
import json

def chat():

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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

        vocab1, word_to_id1, id_to_word1 = cdu.create_vocab(cleaned_questions, cleaned_answers)
        # word_to_id_ = open('word_to_id.txt','r')
        word_to_id = dict()
        id_to_word = dict()

        f=open('word_to_id21.txt', 'r',encoding='utf-8')
        a = f.read()
        word_to_id = eval(a)


        f1=open('id_to_word21.txt', 'r',encoding='utf-8')
        a1 = f1.read()
        id_to_word = eval(a1)


        # with open('word_to_id3.txt', 'r',encoding='utf-8') as f:
        #     for line in f:
        #
        #
        #         try:
        #             id_to_word[int(line.split('\t')[1].strip())] = line.split('\t')[0]
        #         except:
        #             print(line)
        #
        #         try:
        #             word_to_id[line.split('\t')[0]] = int(line.split('\t')[1].strip())
        #         except:
        #             print(line)





        # code1 =word_to_id['\\\\']
        # word_to_id.pop('\\\\')
        # word_to_id['\\']=code1
        # id_to_word[code1] = '\\'
        #
        # code1 =word_to_id['\\n']
        # word_to_id.pop('\\n')
        # word_to_id['\n']=code1
        # id_to_word[code1] = '\n'
        #
        # code1 =word_to_id['￣￣']
        # word_to_id.pop('￣￣')
        # word_to_id['￣']=code1
        # id_to_word[code1] = '￣'


        vocab = list(word_to_id.keys())

        # a = set(word_to_id.keys())
        # b = set(id_to_word.values())
        # a1 = set(word_to_id1.keys())
        # b1 = set(id_to_word1.values())
        # c = set(vocab)
        # c1 = set(vocab1)


        # word_to_id.update(word_to_id1.pop('\\\\'))
        # word_code = word_to_id['好呀']
        # id_to_word1[word_code] = '好呀'

        model = create_model(sess, vocab, word_to_id,id_to_word,
                            forward_only=True)
        # model.batch_size = 1  # We decode one sentence at a time.
        # Load vocabularies.
        # vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        # vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            predicted_sentence = get_predicted_sentence_2(sentence, word_to_id, id_to_word, model, sess)
            print(predicted_sentence)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
