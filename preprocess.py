import os
from urllib import urlretrieve
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

if __name__=='__main__':

    downloaded_data_dir = os.path.join('data', 'download', 'squad')
    processed_data_dir = os.path.join('data', 'processed', 'squad')
    
    print "Downloading the file into {}".format(downloaded_data_dir)
    
    dev_file = 'dev-v1.1.json'
    download_from_url(downloaded_data_dir, squad_url, dev_file, 4854279L)
