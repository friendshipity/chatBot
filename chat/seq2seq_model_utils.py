from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils.model_utils import Chatbot
import utils.config as config
from tensorflow.python.platform import gfile
from utils.cornell_data_utils import bucket_one_question

from utils.config import FLAGS, BUCKETS
from chat import data_utils
from utils.model_utils import Chatbot
from six.moves import xrange


def create_model(session, vocab, word_to_id, id_to_word, forward_only):
    """Create translation model and initialize or load parameters in session."""
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
                    config.BUCKETS)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    # saver = tf.train.Saver(max_to_keep=10)
    if ckpt:
        # and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    # else:
    #   print("Created model with fresh parameters.")
    #   session.run(tf.initialize_all_variables())
    return model


def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids_1(input_sentence, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    outputs = []
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_utils.EOS_ID:
            break
        else:
            outputs.append(selected_token_id)

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])

    return output_sentence


def get_predicted_sentence_1(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids_1(input_sentence, vocab)
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    bucket_data = bucket_one_question(input_token_ids, vocab, bucket_id)
    bucket_data = np.array(bucket_data)


    # Get a 1-element batch to feed the sentence to the model.
    # encoder_inputs, decoder_inputs, target_weights = model.get_batch(input_feed, bucket_id)

    feed_dict = {model.inputs: [bucket_data]*config.BATCH_SIZE,
                 model.keep_probs: config.KEEP_PROBS,
                 model.decoder_seq_len: [BUCKETS[bucket_id][1]]*config.BATCH_SIZE,
                 model.encoder_seq_len: [BUCKETS[bucket_id][0]]*config.BATCH_SIZE}

    # feed_dict = {model.inputs: [bucket_data]*64,
    #              model.keep_probs: config.KEEP_PROBS,
    #              model.decoder_seq_len: [BUCKETS[bucket_id][1]]*64,
    #              model.encoder_seq_len: [BUCKETS[bucket_id][0]]*64}
    #
    preds = sess.run([model.predictions], feed_dict=feed_dict)

    # Get output logits for the sentence.
    # _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    outputs = []
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    # Forming output sentence on natural language
    output_array = preds[0][int(np.random.uniform() * config.BATCH_SIZE)]
    for i in output_array:
        outputs.append(i)
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])

    return output_sentence



def get_predicted_sentence_2(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids_1(input_sentence, vocab)
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    bucket_data = bucket_one_question(input_token_ids, vocab, bucket_id)
    bucket_data = np.array(bucket_data)
    model.batch_size = 1

    # Get a 1-element batch to feed the sentence to the model.
    # encoder_inputs, decoder_inputs, target_weights = model.get_batch(input_feed, bucket_id)

    feed_dict = {model.inputs: [bucket_data]*config.BATCH_SIZE,
                 model.keep_probs: config.KEEP_PROBS,
                 model.decoder_seq_len: [BUCKETS[bucket_id][1]]*config.BATCH_SIZE,
                 model.encoder_seq_len: [BUCKETS[bucket_id][0]]*config.BATCH_SIZE}



    preds = sess.run([model.predictions], feed_dict=feed_dict)

    # Get output logits for the sentence.
    # _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    # Forming output sentence on natural language
    output_array = preds[0]
    output_sentencess = []
    for stc in output_array:
        outputs = []
        for i in stc:
            outputs.append(i)
        rev_vocab[0]=''
        rev_vocab[1]=''
        rev_vocab[2]=''
        rev_vocab[3]=''
        output_sentence = ' '.join([rev_vocab[output] for output in outputs])
        output_sentencess.append(output_sentence)

    # for sentence in output_sentencess:

    return output_sentencess