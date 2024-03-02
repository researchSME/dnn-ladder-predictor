import pickle
import os

def read_pickle(file_path: str):
    with open(file_path, 'rb') as inp:
        return pickle.load(inp)
    
def write_pickle(file_path: str, obj: object):
    with open(file_path, 'wb') as outp:
        pickle.dump(obj, outp)
        
def makedir(file_path: str):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
