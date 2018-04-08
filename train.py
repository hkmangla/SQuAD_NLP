import tensorflow as tf
import sys
sys.path.append('tools/')
from utils import *
from dataword_into_ids import initialize_vocab
from encoder import Encoder
from decoder import Decoder
from qa_system import QASystem

import logging

def set_logger(file_path):
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(file_path,'log.txt'))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

logger = set_logger("log.txt")

def initialize_model(session, model, train_dir){
    ckpt = tf.train.get_checkpoint_state(train_dir)
        
    if ckpt and os.path.exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)

    else:
        logger.info("Initializing model with fresh parameters")
        session.run(tf.global_variables_intializer())
        
    return model
}
def main(_):
    data_dir = os.path.join('data','processed','squad') 
    train_data, val_data = load_data(data_dir, max_context_len, max_question_len)
    
    vocab_path = os.path.join('data','processed','vocab','vocab.dat')
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embd_dir = os.path.join('data','download','glove','glove.trimmered_100.npz')
    embed = load_embeddings(embd_dir)

    emdeddings = tf.constant(embed, tf.float32)
    #Dummy code... must have to edit

    #Below functions are not implemented till
    encoder = Encoder(args)
    decoder = Decoder(args)

    qa = QASystem(encoder, decoder, embeddings, vocab)
    
    train_dir = os.path.join("train")
    #Here training part
    
    with tf.Session() as sess:
        initialize_model(sess, qa, train_dir)
        
        qa.train(sess, train_data, val_data, train_dir)

