import tensorflow as tf
import sys
sys.path.append('/home/hkmangla/workspace/college_project/tools/')
from utils import *
from encoder import Encoder
from decoder import Decoder
from qa_system import QASystem

import logging

logger = set_logger("/home/hkmangla/workspace/college_project/train/log.txt")

model_name = '/baseline4'
tf.app.flags.DEFINE_float("base_lr", 0.001, "Basic learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Decay ratio after a certain batch number.")
tf.app.flags.DEFINE_integer("decay_number", 600, "Anneal learning rate after this number of batches.")
tf.app.flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.4, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 2, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("max_context_len", 600, "max length of the context input")
tf.app.flags.DEFINE_integer("max_question_len", 100, "max length of the question input")
tf.app.flags.DEFINE_string("data_dir", "/home/hkmangla/workspace/college_project/data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "/home/hkmangla/workspace/college_project/train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log" + model_name, "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "/home/hkmangla/workspace/college_project/data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("model_name", model_name, "name of the model")
tf.app.flags.DEFINE_string('summary_dir', '/home/hkmangla/workspace/college_project/summary' + model_name, 'tensorboard summary dir')
tf.app.flags.DEFINE_bool('summary_flag', True, 'if true log summary')


FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
        
    if ckpt and os.path.exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)

    else:
        logger.info("Initializing model with fresh parameters")
        session.run(tf.initialize_all_variables())
        
    return model

def main(_):

    data_dir = os.path.join('/home/hkmangla/workspace/college_project/','data','processed','squad','train') 
    train_data, val_data = load_data(data_dir, FLAGS.max_context_len, FLAGS.max_question_len)
    
    vocab_path = os.path.join('/home/hkmangla/workspace/college_project/','data','processed','vocab','vocab.dat')
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embd_dir = os.path.join('/home/hkmangla/workspace/college_project/','data','download','glove','glove.trimmered_100.npz')
    print embd_dir
    embed = load_embeddings(embd_dir)

    embeddings = tf.constant(embed, tf.float32)
    encoder = Encoder(FLAGS.state_size, FLAGS.summary_flag, FLAGS.max_context_len, FLAGS.max_question_len)
    decoder = Decoder(FLAGS.state_size, FLAGS.summary_flag)

    qa = QASystem(encoder, decoder, FLAGS, embeddings, vocab)
    
    train_dir = os.path.join('/home/hkmangla/workspace/college_project/',"train")
    #Here training part
    
    with tf.Session() as sess:
        initialize_model(sess, qa, train_dir)
        
        qa.train(sess, train_data, val_data, train_dir)

if __name__ == '__main__':
    tf.app.run()
