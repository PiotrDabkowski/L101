import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMBlockFusedCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.layers import fully_connected
from gen_spam_dataset import get_val_bm, get_train_bm
from nice_trainer import NiceTrainer

class GenSpamCharRnn:
    BATCH_SIZE = 32
    SEQ_LEN = 512
    NUM_CHARS = 256

    CHAR_EMBEDDING_SIZE = 64
    HIDDEN_LAYER_SIZE = 1024
    LAYERS = 2
    CHAR_PRED_FC_SIZE = 512
    SPAM_PRED_FC_SIZE = 512

    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.000002

    def __init__(self):
        self.char_input = tf.placeholder(tf.int32, (self.BATCH_SIZE, self.SEQ_LEN), 'char_input')
        char_input_by_time = tf.transpose(self.char_input)

        self.spam_labels = tf.placeholder(tf.int32, (self.BATCH_SIZE,), 'spam_labels')

        out, next_state = self.get_lstm_outputs(char_input_by_time, last_state=None, reuse=False)

        char_raw_scores = self.get_char_predictions(out, reuse=False)
        spam_raw_scores = self.get_spam_predictions(out, reuse=False)
        self.spam_probs = tf.nn.softmax(spam_raw_scores)

        spam_raw_scores2 = self.get_spam_predictions2(out, reuse=False)
        self.spam_probs2 = tf.nn.softmax(spam_raw_scores2)


        self.char_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(char_preds_at_t, char_input_by_time[t+1])) for t, char_preds_at_t in enumerate(tf.unstack(char_raw_scores)[:-1])])
        self.spam_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(spam_raw_scores, self.spam_labels))
        self.spam_loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(spam_raw_scores2, self.spam_labels))


        # now try to collect all the weights for weight decay
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spam_gen_rnn') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='char_predictor') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spam_predictor')
        l2_variables = [e for e in self.all_variables if len(e.get_shape().as_list())>1]

        self.all_variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='char_embedding')
        print 'Num weights', len(l2_variables)
        assert len(l2_variables)==6
        self.l2_loss = self.WEIGHT_DECAY * sum(map(tf.nn.l2_loss, l2_variables))

        self.full_loss = self.char_loss + 11*self.spam_loss + self.l2_loss

        self.train_op = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.full_loss)

        self.train_op2 = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.spam_loss2, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spam_a_predictor'))


    def get_lstm_outputs(self, chars, last_state=None, reuse=False):
        with tf.variable_scope('char_embedding', reuse=reuse):
            self.char_embedding = tf.get_variable('char_embedding', initializer=tf.orthogonal_initializer()(
                (self.NUM_CHARS, self.CHAR_EMBEDDING_SIZE)), dtype=tf.float32)
            out = tf.nn.embedding_lookup(self.char_embedding, chars)
        with tf.variable_scope('spam_gen_rnn', reuse=reuse):
            next_state = []
            for layer in xrange(self.LAYERS):
                with tf.variable_scope('lstm_layer_%d' % layer, initializer=tf.orthogonal_initializer()) as scope:
                    out, next_state_part = LSTMBlockFusedCell(self.HIDDEN_LAYER_SIZE)(out,
                                                                                      last_state[layer] if last_state is not None else None,
                                                                                      dtype=tf.float32,
                                                                                      scope=scope)
                    out = tf.nn.relu(out) # ???? already applied by LSTMBlockFusedCell?
                    next_state.append(next_state_part)
            return out, next_state

    def get_lstm_outputs2(self, chars, last_state=None, reuse=False):
        with tf.variable_scope('char_embedding', reuse=reuse):
            self.char_embedding = tf.get_variable('char_embedding', initializer=tf.orthogonal_initializer()(
                (self.NUM_CHARS, self.CHAR_EMBEDDING_SIZE)), dtype=tf.float32)
            out = tf.nn.embedding_lookup(self.char_embedding, chars)
        with tf.variable_scope('spam_gen_rnn', reuse=reuse):
            cud = CudnnLSTM(self.LAYERS, self.HIDDEN_LAYER_SIZE, self.CHAR_EMBEDDING_SIZE, dropout=0.5)
            out, a, b = cud(out, None, None, {})
        return out, (a, b)

    def get_char_predictions(self, out, reuse=False):  # returns logits
        with tf.variable_scope('char_predictor', reuse=reuse):
            with tf.variable_scope('char_predictor_fc_layer'):
                out = fully_connected(out, self.CHAR_PRED_FC_SIZE)
            with tf.variable_scope('char_predictions'):
                out = fully_connected(out, self.NUM_CHARS, activation_fn=None)
        return out

    def get_spam_predictions(self, out, reuse=False):  # returns logits
        out = tf.reduce_mean(out, 0)
        with tf.variable_scope('spam_predictor', reuse=reuse):
            with tf.variable_scope('spam_predictor_fc_layer'):
                out = fully_connected(out, self.SPAM_PRED_FC_SIZE)
            with tf.variable_scope('spam_predictions'):
                out = fully_connected(out, 2, activation_fn=None)
        return out

    def get_spam_predictions2(self, out, reuse=False):  # returns logits
        out = tf.reduce_mean(out, 0)
        with tf.variable_scope('spam_a_predictor', reuse=reuse):
            with tf.variable_scope('spam_predictor_fc_layer'):
                out = fully_connected(out, self.SPAM_PRED_FC_SIZE)
            with tf.variable_scope('spam_predictions'):
                out = fully_connected(out, 2, activation_fn=None)
        return out



def calculate_spam_acc(extra_vars, batch):
    labels = batch[1]
    probs = extra_vars['spam_probs']
    score = 0
    for l, p in zip(labels, probs):
        if p[l] > 0.5:
            score += 1
    return float(score) / len(labels)



gen_spam = GenSpamCharRnn()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(gen_spam.all_variables)
    ckpt = tf.train.get_checkpoint_state('save')
    epoch = 1
    if ckpt is not None:
        print 'Restoring from', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1

    train_bm = get_train_bm(gen_spam.SEQ_LEN, int(0.85*gen_spam.SEQ_LEN), gen_spam.BATCH_SIZE)
    val_bm = get_val_bm(gen_spam.SEQ_LEN, int(0.85*gen_spam.SEQ_LEN), gen_spam.BATCH_SIZE)

    nt = NiceTrainer(sess, train_bm, [gen_spam.char_input, gen_spam.spam_labels], gen_spam.train_op2, None,
                     {'char_loss': gen_spam.char_loss, 'spam_loss': gen_spam.spam_loss2, 'spam_probs': gen_spam.spam_probs2},
                     printable_vars=['char_loss', 'spam_loss', 'acc'],
                     computed_variables={'acc': calculate_spam_acc})

    for epoch in range(epoch, 100):
        print 'Epoch', epoch
        nt.train()
        #saver.save(sess, 'save/model.tfm', epoch)
        print 'Model saved'


