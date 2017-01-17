# -*- coding: utf-8 -*-
import random, sys, os
print os.environ
import tensorflow as tf
from nice_trainer import NiceTrainer
from gen_spam_dataset import get_val_bm, get_train_bm, load_messages, remove_confusing_tags
import numpy as np
import re

SAVE_DIR = 'save_char_rnn'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


class SpellChecker:
    WIDTH = 512
    NUM_RNN_LAYERS = 3
    CHAR_EMBEDDING_SIZE = 64
    RNN_ACTIVATION = tf.nn.tanh
    CELL_TYPE = tf.contrib.rnn.GRUCell
    __NUM_CHARS = 256
    NUM_TRAIN_SEQ = 150
    BATCH_SIZE = 32
    IDENTITY_TASK_EVERY_N = 300000
    NUM_IDENTITIES = 2
    LEARNING_RATE = 0.001

    def __init__(self, sample=False):
        self.sample = sample
        if self.sample:
            self.BATCH_SIZE = 1
            self.NUM_TRAIN_SEQ = 1
        self.char_id_inp = tf.placeholder(tf.int32, (self.BATCH_SIZE, self.NUM_TRAIN_SEQ), name='char_id_inp')

        self.identity_inp = tf.placeholder(tf.int32, (self.BATCH_SIZE,), name='identity_inp')

        # char embedding
        self.char_embedding = tf.get_variable('char_embedding', initializer=tf.orthogonal_initializer()((self.__NUM_CHARS, self.CHAR_EMBEDDING_SIZE)), dtype=tf.float32)
        self.char_embedding_out = tf.nn.embedding_lookup(self.char_embedding, self.char_id_inp)
        print self.char_embedding_out
        # multi layer rnn
        self.cell = self.CELL_TYPE(self.WIDTH)
        self.rnn_outs = [tf.unstack(self.char_embedding_out, axis=1)]
        self.rnn_state_outs = []
        if sample:
            self.initial_rnn_state_inp = tf.placeholder(tf.float32, (self.NUM_RNN_LAYERS, self.BATCH_SIZE, self.WIDTH), name='initial_rnn_state_inp')
        else:
            self.initial_rnn_state_inp = tf.constant(np.zeros((self.NUM_RNN_LAYERS, self.BATCH_SIZE, self.WIDTH)), dtype=tf.float32, name='initial_rnn_state_inp')
        rnn_layer_states = tf.unstack(self.initial_rnn_state_inp)
        for layer in xrange(self.NUM_RNN_LAYERS):
            with tf.variable_scope('RNN_Layer%d' % (layer+1)):
                rnn_out, state_out = tf.nn.dynamic_rnn(self.cell, tf.stack(self.rnn_outs[layer]), initial_state=rnn_layer_states[layer], time_major=True)
            self.rnn_outs.append(rnn_out)
            self.rnn_state_outs.append(state_out)

        self.rnn_state_out = tf.concat(0, self.rnn_outs[1:])

        # Char prediction task
        # calculate final char predictions
        self.rnn_out_2_probs = tf.get_variable('rnn_out_2_probs', initializer=tf.random_normal((self.WIDTH, self.__NUM_CHARS))/10.0)
        self.rnn_out_2_probs_bias = tf.get_variable('rnn_out_2_probs_bias', initializer=tf.random_normal((1, self.__NUM_CHARS))/10.0)
        self.probs = []
        self.loss = tf.constant(0, dtype=tf.float32)
        for guess in xrange(self.NUM_TRAIN_SEQ):
            prob = tf.nn.softmax(tf.matmul(self.rnn_outs[-1][guess], self.rnn_out_2_probs) + self.rnn_out_2_probs_bias)
            self.probs.append(prob)
            if guess + 1 < self.NUM_TRAIN_SEQ:
                correct_probs = tf.gather_nd(prob, tf.transpose((np.arange(self.BATCH_SIZE), self.char_id_inp[:, guess+1])))
                self.loss += tf.reduce_sum(-tf.log(correct_probs))
        self.loss = self.loss / (self.BATCH_SIZE * self.NUM_TRAIN_SEQ)

        # Identity prediction task
        self.rnn_state_2_identity = tf.get_variable('rnn_state_2_identity', initializer=tf.random_normal((self.WIDTH, self.NUM_IDENTITIES))/10.0)
        self.rnn_state_2_identity_bias = tf.get_variable('rnn_state_2_identity_bias', initializer=tf.random_normal((1, self.NUM_IDENTITIES))/10.0)
        self.identitiy_probs = tf.nn.softmax(tf.matmul(self.rnn_outs[-1][-1], self.rnn_state_2_identity) + self.rnn_state_2_identity_bias)
        correct_identity_probs = tf.gather_nd(self.identitiy_probs, tf.transpose((np.arange(self.BATCH_SIZE), self.identity_inp)))
        self.identity_loss = tf.reduce_mean(-tf.log(correct_identity_probs))

        # quick identiy infer:
        self.quick_identity_infer = tf.nn.softmax(tf.matmul(tf.reshape(self.initial_rnn_state_inp[self.NUM_RNN_LAYERS-1,:,:], (self.BATCH_SIZE, self.WIDTH)), self.rnn_state_2_identity) + self.rnn_state_2_identity_bias)

        self.train_op = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss+self.identity_loss)


    def get_zero_state(self):
        return np.zeros((self.NUM_RNN_LAYERS, self.BATCH_SIZE, self.WIDTH))



    def sample_one(self, sess, char, initial_state, max_choice=False):
        if initial_state is None:
            initial_state = self.get_zero_state()
        probs, end_state = sess.run([self.probs, self.rnn_state_out],
                         {self.initial_rnn_state_inp: initial_state,
                          self.char_id_inp: np.array([char]).reshape(1, 1)})
        if not max_choice:
            return np.random.choice(np.arange(self.__NUM_CHARS), p=probs[0].ravel()), end_state
        else:
            return np.argmax(probs), end_state

    def sample_n(self, sess, n, seed='. ', initial_state=None, sample_until_matches=None):
        if sample_until_matches:
            n = 10000
        if not seed:
            seed = chr(random.randrange(256))
        pred_char = 0
        for c in seed:
            pred_char, initial_state = self.sample_one(sess, ord(c), initial_state)
        pred_seq = [pred_char]
        for _ in xrange(n):
            pred_char, initial_state = self.sample_one(sess, pred_seq[-1], initial_state)
            pred_seq.append(pred_char)
            if sample_until_matches and re.search(sample_until_matches, ''.join(map(chr, pred_seq))):
                break
        spam = sess.run(self.quick_identity_infer, {self.initial_rnn_state_inp: initial_state})
        return seed + ''.join(map(chr, pred_seq)), initial_state, spam[0][1]

    def get_spam_prob(self, sess, message):
        return self.sample_n(sess, 0, seed=message)[2]



def calculate_spam_acc(extra_vars, batch):
    labels = batch[1]
    probs = extra_vars['spam_probs']
    score = 0
    for l, p in zip(labels, probs):
        if p[l] > 0.5:
            score += 1
    return float(score) / len(labels)


gen_spam = SpellChecker(False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    epoch = 1
    if ckpt is not None:
        print 'Restoring from', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1


    train_bm = get_train_bm(gen_spam.NUM_TRAIN_SEQ, int(0.85*gen_spam.NUM_TRAIN_SEQ), gen_spam.BATCH_SIZE)
    val_bm = get_val_bm(gen_spam.NUM_TRAIN_SEQ, int(0.85*gen_spam.NUM_TRAIN_SEQ), gen_spam.BATCH_SIZE)

    nt = NiceTrainer(sess, train_bm, [gen_spam.char_id_inp, gen_spam.identity_inp], gen_spam.train_op, val_bm,
                     {'char_loss': gen_spam.loss, 'spam_loss': gen_spam.identity_loss, 'spam_probs': gen_spam.identitiy_probs},
                     printable_vars=['char_loss', 'spam_loss', 'acc'],
                     computed_variables={'acc': calculate_spam_acc})

    for epoch in range(epoch, 100):
        print 'Epoch', epoch
        nt.train()
        saver.save(sess, os.path.join(SAVE_DIR, 'model.tfm'), epoch)
        print 'Model saved'

