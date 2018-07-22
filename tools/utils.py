import logging
import os
import numpy as np
import time
import tensorflow as tf

def set_logger(file_path):
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, "r") as f:
            for line in f:
                rev_vocab.append(line.strip('\n'))
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    else:
        return ValueError("File {} does not found".format(vocab_path))


def read(data_file, size):
    count = 0
    data = []

    with open(data_file,'r') as f:
        for line in f:
            data.append(list(map(lambda x: int(x), line.strip().split())))
            count += 1

            if size is not None:
                if count >= size:
                    break

    return data

def pad_sequence(data, max_len):
    mask = []
    pad_vect = []
    for sentence in data:
        if len(sentence) >= max_len:
            pad_vect.append(sentence[:max_len])
            mask.append([True]*max_len)
        else:
            pad_vect.append(sentence + [0]*(max_len - len(sentence)))
            mask.append([True]*len(sentence) + [False]*(max_len - len(sentence)))

    return pad_vect, mask

def variable_summaries(var, name_scope, matrix = True):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

  with tf.name_scope(name_scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    if matrix:
        norm = tf.sqrt(tf.reduce_sum(var * var))
        tf.summary.scalar('norm', norm)

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

def load_data(data_dir, max_context_len, max_question_len, size=100):

    train_context = read(os.path.join(data_dir, 'train.ids.contexts'), size)
    train_question = read(os.path.join(data_dir, 'train.ids.questions'), size)
    train_span = read(os.path.join(data_dir, 'train.spans'), size)

    train_context_data, train_context_mask = pad_sequence(train_context,max_context_len)
    train_question_data, train_question_mask = pad_sequence(train_question, max_question_len)
    start_span_vector_train, end_span_vector_train = process_span(train_span, max_context_len)
    
    train_data = vectorize(train_context_data, train_context_mask, train_question_data,
                          train_question_mask, start_span_vector_train ,end_span_vector_train, train_span)

    val_context = read(os.path.join(data_dir, 'val.ids.contexts'), size)
    val_question = read(os.path.join(data_dir, 'val.ids.questions'), size)
    val_span = read(os.path.join(data_dir, 'val.spans'), size)

    val_context_data, val_context_mask = pad_sequence(val_context,max_context_len)
    val_question_data, val_question_mask = pad_sequence(val_question, max_question_len)
    start_span_vector_val, end_span_vector_val = process_span(val_span, max_context_len)
    
    val_data = vectorize(val_context_data, val_context_mask, val_question_data,
                          val_question_mask, start_span_vector_val, end_span_vector_val, val_span)
    
    return train_data, val_data

def vectorize(*args):
    return list(zip(*args))

def load_embeddings(emd_dir):
    return np.load(emd_dir)['glove']

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:
        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...
    Or with multiple data sources:
        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...
    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.
    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

def print_sentence(output, sentence, labels, predictions):

    spacings = [max(len(sentence[i]), len(labels[i]), len(predictions[i])) for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y*: ")
    for token, spacing in zip(labels, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
        output.write("\n")