import tensorflow as tf

VOCAB_THRESHOLD = 10


BUCKETS = [(10, 15), (15, 25), (25, 45), (45, 60), (60, 100)] #First try buckets you can tweak these

EPOCHS = 500

BATCH_SIZE = 64

RNN_SIZE = 512 #128

NUM_LAYERS = 3

ENCODING_EMBED_SIZE = 512
DECODING_EMBED_SIZE = 512

LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9 #nisam siguran da cu ovo koristiti
MIN_LEARNING_RATE = 0.0001

KEEP_PROBS = 0.5

CLIP_RATE = 4




# TEST_DATASET_PATH = 'tf_seq2seq_chatbot/data/test/test_set.txt'
# SAVE_DATA_DIR = '/var/lib/tf_seq2seq_chatbot/'
# SAVE_DATA_DIR = 'D:/Git/tf_seq2seq_chatbot/tf_seq2seq_chatbot/lib/tf_seq2seq_chatbot/'
SAVE_DATA_DIR = 'C:/Users/Administrator/PycharmProjects/test/chatBot/'
MODEL_DIR = 'C:/Users/Administrator/PycharmProjects/test/chatBot/checkpoint/'

# tf.app.flags.DEFINE_string('data_dir', SAVE_DATA_DIR + 'data', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', MODEL_DIR , 'Train directory')
# tf.app.flags.DEFINE_string('results_dir', SAVE_DATA_DIR + 'results', 'Train directory')

tf.app.flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', LEARNING_RATE_DECAY, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size to use during training.')

# tf.app.flags.DEFINE_integer('vocab_size', 20000, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('size', RNN_SIZE, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', NUM_LAYERS, 'Number of layers in the model.')

# tf.app.flags.DEFINE_integer('max_train_data_size', 12000, 'Limit on the size of training data (0: no limit).')
# tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]