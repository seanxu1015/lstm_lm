# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:06:11 2016

@author: sean
"""


import sys, time
import numpy as np
from utils import get_ptb_dataset, Vocab
from utils import ptb_iterator
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel
#from q2_initialization import xavier_weight_init

class Config(object):
    """Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    batch_size = 64
    embed_size = 50
    hidden_size = 100
    num_steps = 10
    max_epochs = 16
    early_stopping = 2
    dropout = 0.5
    lr = 0.001
    #l2 = 0.001
    
class LSTM_Model(LanguageModel):
    
    def load_data(self, debug=False):
        
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encode_train = np.array([self.vocab.encode(word) \
                                      for word in get_ptb_dataset('train')], dtype=np.int32)
        self.encode_valid = np.array([self.vocab.encode(word) \
                                      for word in get_ptb_dataset('valid')], dtype=np.int32)
        self.encode_test = np.array([self.vocab.encode(word) \
                                     for word in get_ptb_dataset('test')], dtype=np.int32)
                
        if debug:
            num_debug = 1024
            self.encode_train = self.encode_train[:num_debug]
            self.encode_valid = self.encode_valid[:num_debug]
            self.encode_test = self.encode_test[:num_debug]
    
    def add_placeholders(self):
        
        self.input_placeholder = tf.placeholder(dtype=tf.int32, \
                                                shape=[None, self.config.num_steps], name='Input')
        self.label_placeholder = tf.placeholder(dtype=tf.int32, \
                                                shape=[None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name='Dropout')
    
    
    def add_embedding(self):
        
        embedding = tf.get_variable('Embedding', \
                                    shape=[len(self.vocab), self.config.embed_size], \
                                    dtype=tf.float32, trainable=True)
        inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)        
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]
        return inputs
        
    
    def add_projections(self, rnn_outputs):
        
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix', shape=[self.config.hidden_size, len(self.vocab)])
            b_2 = tf.get_variable('Proj_b', shape=[len(self.vocab)])
            outputs = [tf.matmul(o, U) + b_2 for o in rnn_outputs]
            
        return outputs

    
    def add_loss_op(self, outputs):
        
        all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        cross_entropy = sequence_loss([outputs],\
                                      [tf.reshape(self.label_placeholder, [-1])],\
                                      all_ones, len(self.vocab))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
                
        return loss
    
    
    def add_train_op(self, loss):
        
        optimizor = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizor.minimize(loss)
        
        return train_op
    
    
    def add_model(self, inputs):
        
        #with tf.variable_scope('InputDropout'):
        #    inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]
        
        with tf.variable_scope('LSTM') as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            self.initial_cell = tf.zeros([self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            cell = self.initial_cell
            outputs = []
            cells = []
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                
                current_input = tf.nn.dropout(current_input, self.dropout_placeholder)                
                
                W_I = tf.get_variable('W_I', shape=[self.config.embed_size, \
                                                    self.config.hidden_size])
                U_I = tf.get_variable('U_I', shape=[self.config.hidden_size, \
                                                    self.config.hidden_size])
                b_I = tf.get_variable('b_I', shape=[self.config.hidden_size])
                I = tf.nn.relu(tf.matmul(current_input, W_I) + tf.matmul(state, U_I) + b_I)
                
                W_f = tf.get_variable('W_f', shape=[self.config.embed_size, \
                                                    self.config.hidden_size])
                U_f = tf.get_variable('U_f', shape=[self.config.hidden_size, \
                                                    self.config.hidden_size])
                b_f = tf.get_variable('b_f', shape=[self.config.hidden_size])
                f = tf.nn.relu(tf.matmul(current_input, W_f) + tf.matmul(state, U_f) + b_f)
                
                W_o = tf.get_variable('W_o', shape=[self.config.embed_size, \
                                                    self.config.hidden_size])
                U_o = tf.get_variable('U_o', shape=[self.config.hidden_size, \
                                                    self.config.hidden_size])
                b_o = tf.get_variable('b_o', shape=[self.config.hidden_size])
                o = tf.nn.relu(tf.matmul(current_input, W_o) + tf.matmul(state, U_o) + b_o)
                
                W_c = tf.get_variable('W_c', shape=[self.config.embed_size, \
                                                    self.config.hidden_size])
                U_c = tf.get_variable('U_c', shape=[self.config.hidden_size, \
                                                    self.config.hidden_size])
                b_c = tf.get_variable('b_c', shape=[self.config.hidden_size])
                c_hat = tf.nn.relu(tf.matmul(current_input, W_c) + tf.matmul(state, U_c) + b_c)
                
                cell = f * cell + I * c_hat
                state = o * tf.nn.tanh(cell)
                state = tf.nn.dropout(state, self.dropout_placeholder)
                cells.append(cell)
                outputs.append(state)
            self.final_state = outputs[-1]
            
        return outputs
        
        
    
    def __init__(self, config):
        
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projections(self.rnn_outputs)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        outputs = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
        
        self.calculate_loss = self.add_loss_op(outputs)
        self.train_step = self.add_train_op(self.calculate_loss)
        
        
    
    def run_epoch(self, session, data, train_op=None, verbose=10):
        
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(
            ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.label_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}
            loss, state, _ = session.run(
                [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))
    

def test_LSTM():
    config = Config()
    #gen_config = deepcopy(config)
    #gen_config.batch_size = gen_config.num_steps = 1
    
    with tf.variable_scope('LSTM_LM') as scope:
        model = LSTM_Model(config)
        scope.reuse_variables()
        
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        
        session.run(init)
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            
            model.run_epoch(session, model.encode_train, train_op=model.train_step)
            valid_pp = model.run_epoch(session, model.encode_valid)
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_lstmlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)


if __name__ == '__main__':
    test_LSTM()
    
    
