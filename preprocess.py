import os
from urllib import urlretrieve
import json
from tqdm import tqdm
squad_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'

def reporthook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):

        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        
    return inner

def download_from_url(download_dir, download_url, file_name, file_size=None):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    full_file_name = os.path.join(download_dir, file_name)
    
    if not os.path.exists(full_file_name):
        try:
            print "\nDownloading file {}".format(download_url + file_name)
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
                local_filename, _ = urlretrieve(download_url + file_name, full_file_name, reporthook=reporthook(t))

        except AttributeError as e:
            print "An error occured during downloading the file! Please get the dataset using browser or try again.."
            raise e
    
    #check if the downloaded file has the same size of file or not
    file_stats = os.stat(full_file_name)

    if file_size is None or file_stats.st_size == file_size:
        print "{} is downloaded successfully!".format(file_name)
    else:
        os.remove(full_file_name)
        raise Exception("Unexpected datasize of the file! Please get the file using browser of try again..")

    return full_file_name

def read_json(file_dir, file_name):
    with open(os.path.join(file_dir, file_name), 'r') as json_file:
        data_loaded = json.load(json_file)
    return data_loaded

def convert_data(data, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    contexts_filename = os.path.join(directory, 'contexts')
    questions_filename = os.path.join(directory, 'questions')
    answers_filename = os.path.join(directory, 'answers')
    span_filename = os.path.join(directory, 'spans')
    
    qn = 0
    an = 0
    skipped = 0
    typ = directory.split('/')[-1]
    with open(contexts_filename, 'w+') as context_file, open(questions_filename, 'w+') as question_file, open(answers_filename,'w+') as answer_file, open(span_filename, 'w+') as span_file:

        for articleId in tqdm(range(len(data['data'])), desc="Preprocessing {}".format(typ)):
            articles  = data['data'][articleId]['paragraphs']
            
            for contextId in range(len(articles)):
                context = articles[contextId]['context']
                qas = articles[contextId]['qas']
                
                for questionId in range(len(qas)):
                    question = qas[questionId]['question']
                    answers = qas[questionId]['answers']
                    
                    qn += 1
                    for answerId in range(1):
                        answer_txt = answers[answerId]['text']

                        answer_start = answers[answerId]['answer_start']
                        answer_end = answer_start + len(answer_txt)
                        
                        try:
                            check = context[answer_end]
                            context_file.write(context + '\n')
                            question_file.write(question + '\n')
                            answer_file.write(answer_txt + '\n')
                            span_file.write("( " + str(answer_start) + " " + str(answer_end) +  ")\n") 

                        except Exception as e:
                            skipped += 1
                        
                        an += 1
    print "Total questions skipped in {} is {}".format(typ, skipped)
    return qn,an

def preprocess(download_dir, process_dir, filename, filesize, tier):

    download_from_url(download_dir, squad_url, filename, filesize)
    data = read_json(download_dir, filename)

    num_ques, num_ans = convert_data(data, os.path.join(process_dir, tier))

    return num_ques, num_ans

if __name__=='__main__':

    downloaded_data_dir = os.path.join('data', 'download', 'squad')
    processed_data_dir = os.path.join('data', 'processed', 'squad')
    
    print "Downloading the file into {}".format(downloaded_data_dir)
    
    dev_file = 'dev-v1.1.json'
    train_file = 'train-v1.1.json'

    #Downloading and preprocessing dev data
    num_ques_dev, _ = preprocess(downloaded_data_dir, processed_data_dir, dev_file, 4854279L, 'dev')
    print "Total dev questions {}".format(num_ques_dev)

    #Downloading and preprocessing train data
    num_ques_train, _ = preprocess(downloaded_data_dir, processed_data_dir, train_file, 30288272L, 'train')
    print "Total train questions {}".format(num_ques_train)
