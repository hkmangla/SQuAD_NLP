import os
import nltk
import tensorflow as tf
import numpy as np
import sys
from gtts import gTTS
import re
from train import FLAGS
from encoder import Encoder
from decoder import Decoder
from qa_system import QASystem
sys.path.append('/home/hkmangla/workspace/college_project/tools/')
from utils import *
import logging
from speech_to_text import recog

logger = set_logger("/home/hkmangla/workspace/college_project/train/log.txt")
reload(sys)
sys.setdefaultencoding('utf8')


FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)

    else:
        logger.info("Initializing model with fresh parameters")
        session.run(tf.initialize_all_variables())
        
    return model


def tokenize(sentence):
    tokens = [token for token in nltk.word_tokenize(sentence)]
    return map(lambda x : x.encode('utf8'), tokens)

def tokenize_again(sentence):
    tokens = []
    for word in sentence:
        tokens.extend(re.split(" ", word))

    return [w for w in tokens if w]

def sentence_to_ids(sentence, vocab):
    words = tokenize_again(sentence)

    return [vocab.get(w, 'UNK_ID') for w in words]

undefined_words = [117193,76525,76541,76711,76716,64260,68430,64304,64324,87640,87641,87650,87662,87669]
def take_input(vocab, context):
    
    if context == None:
        context = raw_input("Input the paragraph without pressing <enter> in the middle of paragraph\n>> ")
    
    print("Input the question!")
    inp_choice = raw_input("Do you want to type the question or ask it verbally (T/V)? ")
    question = ''
    try:
        if inp_choice.lower() == 'v':
            question = recog()#raw_input("Input the question\n>> ")
        else:
            question = raw_input("Type the question: ")
    except:
        print("\nNo Internet Connectivity!\n")
        question = raw_input("Please! Type the question: ")
        

    print "You asked: ", str(question)
    context_tokens = tokenize(context)
    question_tokens = tokenize(question)

    context_ids = sentence_to_ids(context_tokens, vocab)
    question_ids = sentence_to_ids(question_tokens, vocab)

    cur_idx = 0
    word_dict = {}

    for i in range(len(context_ids)):
        if context_ids[i] == 'UNK_ID':
            if context_tokens[i] in word_dict.keys():
                context_ids[i] = undefined_words[word_dict[context_tokens[i]]]
            else:
                context_ids[i] = undefined_words[cur_idx]
                word_dict[context_tokens[i]] = cur_idx
                cur_idx += 1

    for i in range(len(question_ids)):
        if question_ids[i] == 'UNK_ID':
            if question_tokens[i] in word_dict.keys():
                question_ids[i] = undefined_words[word_dict[question_tokens[i]]]
            else:
                question_ids[i] = undefined_words[cur_idx]
                word_dict[question_tokens[i]] = cur_idx
                cur_idx += 1

    context_ids = [context_ids]
    question_ids = [question_ids]

    context_ids_ext, context_ids_mask = pad_sequence(context_ids, FLAGS.max_context_len)
    question_ids_ext, question_ids_mask = pad_sequence(question_ids, FLAGS.max_question_len)

    input_data = vectorize(context_ids_ext, context_ids_mask, question_ids_ext, question_ids_mask)

    batch = minibatches(input_data, 1)

    return context_tokens, context, context_ids[0], batch

def speak(mytext):
    language = 'en'
 
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3") 
    os.system("play welcome.mp3")


def main(_):
    vocab, rev_vocab = initialize_vocab('/home/hkmangla/workspace/college_project/data/processed/vocab/vocab.dat')
    embed_path = os.path.join('/home/hkmangla/workspace/college_project', 'data','download','glove','glove.trimmered_100.npz')
    embedding = tf.constant(load_embeddings(embed_path), dtype = tf.float32)
    
    encoder = Encoder(FLAGS.state_size, FLAGS.summary_flag, FLAGS.max_context_len, FLAGS.max_question_len)
    decoder = Decoder(FLAGS.state_size, FLAGS.summary_flag)

    qa = QASystem(encoder, decoder, FLAGS, embedding, rev_vocab)
    with tf.Session() as sess:  
        initialize_model(sess, qa, FLAGS.train_dir)
        context = None
        while True:
            context_tokens, context, context_ids, batch = take_input(vocab, context)
            for i in batch:
                print('Finding the answer....')
                a_s_vec, a_e_vec = qa.answer(sess, i)
                answer_text = qa.formulate_answer(undefined_words, context_tokens, context_ids, rev_vocab, min(a_s_vec, a_e_vec), max(a_s_vec, a_e_vec))
                try:
                    print 'Answer: ', answer_text
                    speak(answer_text)
                except:
                    print "Can't speak the answer due to internet problem!"
            
            choice = raw_input("Do you want to ask another question (y/n): ")
            if choice.lower() == 'n':
                print("Exiting the program.....")
                break 

if __name__ == '__main__':
    tf.app.run()
