import tensorflow as tf
from tensorflow.python.layers.core import Dense
import random
import chat.data_utils as data_utils
import numpy as np

assert tf.__version__ == '1.4.0'


def grap_inputs():
    '''
		This function is used to define all tensorflow graph placeholders (inputs to the TF graph)

		Inputs: None

		Outputs:
			inputs - questions in the case of a Chatbot with dimensions of None, None = batch_size, questions_length
			targets - answers in the case of a Chatbot with dimensions of None, None = batch_size, answers_length
			keep_probs - probabilities used in dropout layer

			encoder_seq_len -  vector which is used to define lenghts of each sample in the inputs to the model
			decoder_seq_len - vector which is used to define lengths of each sample in the targets to the model
			max_seq_len - target sample with the most words in it

    '''

    inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
    targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
    keep_probs = tf.placeholder(tf.float32, name='dropout_rate')
    encoder_seq_len = tf.placeholder(tf.int32, (None,), name='encoder_seq_len')
    decoder_seq_len = tf.placeholder(tf.int32, (None,), name='decoder_seq_len')
    max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')

    return inputs, targets, keep_probs, encoder_seq_len, decoder_seq_len, max_seq_len


def encoder(inputs, rnn_size, number_of_layers, encoder_seq_len, keep_probs, encoder_embed_size, encoder_vocab_size):
    '''
        Used to define encoder of the seq2seq model (The encoder is made of simple dynamic RNN network).

        Inputs:
            inputs -
            rnn_siz - number of units in the RNN layer
            number_of_layer - number of RNN layers that the model uses
            encoder_seq_len - vector of lengths (got from placeholder)
            keep_probs - dropout rate
            encoder_embed_size - size of embedding vector for encoder part
            encoder_vocab_size - number of different words that the model uses in a vocabulary

        Outputs:
            encoder_outputs -
            encoder_states - internal states from the RNN layer(s)

    '''

    def cell(units, rate):
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, rate)

    encoder_cell = tf.contrib.rnn.MultiRNNCell([cell(rnn_size, keep_probs) for _ in range(number_of_layers)])

    encoder_embedings = tf.contrib.layers.embed_sequence(inputs, encoder_vocab_size,
                                                         encoder_embed_size)  # used to create embeding layer for the encoder

    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell,
                                                        encoder_embedings,
                                                        encoder_seq_len,
                                                        dtype=tf.float32)
    return encoder_outputs, encoder_states


def decoder_inputs_preprocessing(targets, word_to_id, batch_size):
    '''
        Helper function used to prepare decoder inputs

        Inputs:
            targets -
            word_to_id - dictionery that the model uses to map each word to it's int representation
            batch_size - number of samples that we put through the model at onces

        Outputs:
            preprocessed version of decoder inputs

    '''

    endings = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1,
                                                                   1])  # This line is used to REMOVE last member of each sample in the decoder_inputs batch
    return tf.concat([tf.fill([batch_size, 1], word_to_id['<GO>']), endings],
                     1)  # returning line and in this line we concat '<GO>' tag at the beginning of each sample in the batch


def decoder(decoder_inputs, enc_states, dec_cell, decoder_embed_size, vocab_size,
            dec_seq_len, max_seq_len, word_to_id, batch_size):
    '''

        The decoder core function.

        Inputs:
            decoder_inputs -
            enc_states - states created by the encoder part of the seq2seq network
            dec_cell - RNN cell used in the decoder RNN (can be attention cell as well)
            decoder_embed_size - vector size of the decoder embedding layer
            vocab_size - number of different words used in the decoder part
            dec_seq_len - vector of lengths for the decoder, obtained from the placeholder
            max_seq_len - sample with max number of words (got from placeholder)
            word_to_id - python dict used to encode each word to it's int representation
            batch_size - number of samples that we put through the model at onces

        Outputs:
            train_dec_outputs -
            inference_dec_output - Inportant for testing and production use!

    '''

    # Defining embedding layer for the Decoder


    embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
    embedings = tf.nn.embedding_lookup(embed_layer, decoder_inputs)

    # Creating Dense (Fully Connected) layer at the end of the Decoder -  used for generating probabilities for each word in the vocabulary
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))

    with tf.variable_scope('decoder'):
        # Training helper used only to read inputs in the TRAINING stage
        train_helper = tf.contrib.seq2seq.TrainingHelper(embedings,
                                                         dec_seq_len)

        # Defining decoder - You can change with BeamSearchDecoder, just beam size
        train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        train_helper,
                                                        enc_states,
                                                        output_layer)

        # Finishing the training decoder
        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_seq_len)

    with tf.variable_scope('decoder',
                           reuse=True):  # we use REUSE option in this scope because we want to get same params learned in the previouse 'decoder' scope
        # getting vector of the '<GO>' tags in the int representation
        starting_id_vec = tf.tile(tf.constant([word_to_id['<GO>']], dtype=tf.int32), [batch_size],
                                  name='starting_id_vec')

        # using basic greedy to get next word in the inference time (based only on probs)
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer,
                                                                    starting_id_vec,
                                                                    word_to_id['<EOS>'])

        # Defining decoder - for inference time
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            enc_states,
                                                            output_layer)

        inference_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_seq_len)

    return train_dec_outputs, inference_dec_output


def vocab_dict(Vocab_filename):
    train_enc_word_to_id = dict()
    train_enc_id_to_word = dict()
    index = 0
    with open(Vocab_filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line
            train_enc_word_to_id[word] = index
            train_enc_id_to_word[index] = word
            index += index
    return train_enc_word_to_id, train_enc_id_to_word


def attention_mech(rnn_size, keep_probs, encoder_outputs, encoder_states, encoder_seq_len, batch_size):
    '''
		The helper function used to create attention mechanism in TF 1.4

		Inputs:
			rnn_size - number of units in the RNN layer
			keep_probs -  dropout rate
			encoder_outputs - ouputs got from the encoder part
			encoder_states - states trained/got from encoder
			encoder_seq_len - 
			batch_size - 

		Outputs:
			dec_cell - attention based decoder cell
			enc_state_new -new encoder stated with attention for the decoder

    '''

    # using internal function to easier create RNN cell
    def cell(units, probs):
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, probs)

    # defining rnn_cell
    decoder_cell = cell(rnn_size, keep_probs)

    # using helper function from seq2seq sub_lib for Bahdanau attention
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                               encoder_outputs,
                                                               encoder_seq_len)

    # finishin attention with the attention holder - Attention Wrapper
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                   attention_mechanism,
                                                   rnn_size / 2)

    # Here we are usingg zero_state of the LSTM (in this case) decoder cell, and feed the value of the last encoder_state to it
    attention_zero = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    enc_state_new = attention_zero.clone(cell_state=encoder_states[-1])

    return dec_cell, enc_state_new


def opt_loss(outputs, targets, dec_seq_len, max_seq_len, learning_rate, clip_rate):
    '''

		Function used to define optimizer and loss function

		Inputs:
			outputs - outputs got from decoder part of the network
			targets - expected outputs/ labels
			dec_seq_len -
			max_seq_len - 
			learning_rate - small nubmer used to decrease value of gradients used to update our network
			clip_rate - tolerance boundries for clipping gradients

		Outputs:
			loss -
			trained_opt - optimizer with clipped gradients
    '''
    logits = tf.identity(outputs.rnn_output)

    mask_weigts = tf.sequence_mask(dec_seq_len, max_seq_len, dtype=tf.float32)

    with tf.variable_scope('opt_loss'):
        # using sequence_loss to optimize the seq2seq model
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                targets,
                                                mask_weigts)

        # Define optimizer
        opt = tf.train.AdamOptimizer(learning_rate)

        # Next 3 lines used to clip gradients {Prevent gradient explosion problem}
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_rate)
        traiend_opt = opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

    return loss, traiend_opt


class Chatbot(object):
    def __init__(self, learning_rate, batch_size, enc_embed_size, dec_embed_size, rnn_size,
                 number_of_layers, vocab_size, word_to_id, id_to_word, clip_rate, buckets):
        # tf.reset_default_graph()

        self.inputs, self.targets, self.keep_probs, self.encoder_seq_len, self.decoder_seq_len, max_seq_len = grap_inputs()
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.buckets = buckets
        self.batch_size = batch_size
        enc_outputs, enc_states = encoder(self.inputs,
                                          rnn_size,
                                          number_of_layers,
                                          self.encoder_seq_len,
                                          self.keep_probs,
                                          enc_embed_size,
                                          vocab_size)

        dec_inputs = decoder_inputs_preprocessing(self.targets,
                                                  word_to_id,
                                                  batch_size)

        decoder_cell, encoder_states_new = attention_mech(rnn_size,
                                                          self.keep_probs,
                                                          enc_outputs,
                                                          enc_states,
                                                          self.encoder_seq_len,
                                                          batch_size)

        train_outputs, inference_output = decoder(dec_inputs,
                                                  encoder_states_new,
                                                  decoder_cell,
                                                  dec_embed_size,
                                                  vocab_size,
                                                  self.decoder_seq_len,
                                                  max_seq_len,
                                                  word_to_id,
                                                  batch_size)

        self.predictions = tf.identity(inference_output.sample_id, name='preds')

        self.loss, self.opt = opt_loss(train_outputs,
                                       self.targets,
                                       self.decoder_seq_len,
                                       max_seq_len,
                                       learning_rate,
                                       clip_rate)
        self.saver = tf.train.Saver(tf.all_variables())

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
    
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
    
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
    
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    # def step(self, session, encoder_inputs, decoder_inputs, target_weights,
    #          bucket_id, forward_only):
    #
    #     """Run a step of the model feeding the given inputs.
    #
    #     Args:
    #       session: tensorflow session to use.
    #       encoder_inputs: list of numpy int vectors to feed as encoder inputs.
    #       decoder_inputs: list of numpy int vectors to feed as decoder inputs.
    #       target_weights: list of numpy float vectors to feed as target weights.
    #       bucket_id: which bucket of the model to use.
    #       forward_only: whether to do the backward step or only forward.
    #
    #     Returns:
    #       A triple consisting of gradient norm (or None if we did not do backward),
    #       average perplexity, and the outputs.
    #
    #     Raises:
    #       ValueError: if length of encoder_inputs, decoder_inputs, or
    #         target_weights disagrees with bucket size for the specified bucket_id.
    #     """
    #     # Check if the sizes match.
    #     encoder_size, decoder_size = self.buckets[bucket_id]
    #     if len(encoder_inputs) != encoder_size:
    #         raise ValueError("Encoder length must be equal to the one in bucket,"
    #                          " %d != %d." % (len(encoder_inputs), encoder_size))
    #     if len(decoder_inputs) != decoder_size:
    #         raise ValueError("Decoder length must be equal to the one in bucket,"
    #                          " %d != %d." % (len(decoder_inputs), decoder_size))
    #     if len(target_weights) != decoder_size:
    #         raise ValueError("Weights length must be equal to the one in bucket,"
    #                          " %d != %d." % (len(target_weights), decoder_size))
    #
    #     # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    #     input_feed = {}
    #     for l in range(encoder_size):
    #         input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    #     for l in range(decoder_size):
    #         input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
    #         input_feed[self.target_weights[l].name] = target_weights[l]
    #
    #     # Since our targets are decoder inputs shifted by one, we need one more.
    #     last_target = self.decoder_inputs[decoder_size].name
    #     input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    #
    #     # Output feed: depends on whether we do a backward step or not.
    #     if not forward_only:
    #         output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
    #                        self.gradient_norms[bucket_id],  # Gradient norm.
    #                        self.losses[bucket_id]]  # Loss for this batch.
    #     else:
    #         output_feed = [self.losses[bucket_id]]  # Loss for this batch.
    #         for l in range(decoder_size):  # Output logits.
    #             output_feed.append(self.outputs[bucket_id][l])
    #
    #     outputs = session.run(output_feed, input_feed)
    #     if not forward_only:
    #         return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    #     else:
    #         return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
