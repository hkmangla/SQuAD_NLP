import os
import numpy as np
import linecache

def save_file(tier, path, indices):
    
    if os.path.exists(os.path.join(path,tier + '.contexts')):
        return None;
    files = ['contexts', 'questions', 'answers', 'spans', 'ids.contexts', 'ids.questions']
    for filename in files:
     with open(os.path.join(path, tier +'.' + filename), 'w') as fp:
         for i in indices:
             fp.write(linecache.getline(os.path.join(path, filename), i))


def train_val_split_tokens(train_dir):
    
    with open(train_dir + "/contexts", "r") as fp:
        total_len = sum(1 for line in fp)
    indices = range(total_len)
    
    val_indices = indices[int(total_len*0.9)::]
    train_indices = indices[:int(total_len*0.9)]
    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)

    save_file('train', train_dir, train_indices)
    save_file('val', train_dir, val_indices)
    
if __name__ == '__main__':
    
    train_dir = os.path.join('data','processed','squad','train')

    train_val_split_tokens(train_dir)
