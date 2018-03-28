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

def load_data(data_dir, max_context_len, max_question_len, size):

    train_context = read(os.path.join(data_dir, 'train/ids.contexts'), size)
    train_question = read(os.path.join(data_dir, 'train/ids.questions'), size)
    train_span = read(os.path.join(data_dir, 'train/spans'), size)


