import hashlib

import os
def file_hash(filename):
    with open(filename,'rb') as f:
        return md5(f.read()).hexdigest()

os.getcwd()

files_list = os.listdir(r'dataset/Dataset')
print (len(files_list))

duplicates=[]
hash_keys=dict()
for index, filename in enumerate(files_list):
    with open('dataset/Dataset' + '/' + filename, 'rb') as f:
        filehash = hashlib.md5(f.read()).hexdigest()
    if filehash not in hash_keys:
        hash_keys[filehash]=index
    else:
        duplicates.append((index,hash_keys[filehash]))
print(duplicates)
for index in duplicates:
    os.remove(files_list[index[0]])