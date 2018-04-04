import os

def read(data_file, size):
    count = 0
    data = []

    with open(data_file,'r') as f:
        for line in f:
            data.append(list(map(lambda x: int(x), line.strip.split())))
            count += 1

            if size is not None:
                if count >= size
                    break

    return data

def pad_sequence(data, max_len):
    mask = []
    pad_vec = []
    for sentence in data:
        if len(vect) >= max_len:
            pad_vect = sentence[:max_len]
            mask = [True]*max_len
        else:
            pad_vect = sentence + [0]*(max_len - len(sentence))
            mask = [True]*len(sentence) + [False]*(max_len - len(sentence)

    return pad_vect, mask

def process_span(span_data, max_context_len):
    start_span_vector = []
    end_span_vector = []
    
    for span in span_data:
        start = [0]*max_context_len
        end = [0]*max_context_len
        if span[0] < max_context_len:
            start[span[0]] = 1
        if span[1] < max_context_len:
            end[span[1]] = 1
        
        start_span_vector.append(start)
        end_span_vector.append(end)

    return start_span_vector, end_span_vector

def load_data(data_dir, max_context_len, max_question_len, size):

    train_context = read(os.path.join(data_dir, 'train/ids.contexts'), size)
    train_question = read(os.path.join(data_dir, 'train/ids.questions'), size)
    train_span = read(os.path.join(data_dir, 'train/spans'), size)

    train_context_data, train_context_mask = pad_sequence(train_context,max_context_len)
    train_qustion_data, train_question_mask = pad_sequence(train_question, max_question_len)
    star_span_vector, end_span_vector = process_span(train_span, max_context_len)
    

def load_embeddings(emd_dir):
    return np.load(dir)['glove']

