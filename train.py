import tensorflow as tf
import sys
sys.path.append('tools/')
from utils import *
from dataword_into_ids import initialize_vocab
from encoder import Encoder
from decoder import Decoder
from qa_system import QASystem


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

    #Here training part
    #TODO


