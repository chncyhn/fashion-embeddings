import os
from pdb import set_trace

def list_files(path, pattern, recursive=True):
    #set_trace()
    pth = os.path.join(os.getcwd(), path)
    ret = []
    if recursive:
        for directory, _, files in os.walk(pth):
            ret.extend([os.path.join(directory,f) for f in files if not pattern or pattern in f])
    else:
        ret = [os.path.join(pth,f) for f in os.listdir(pth) if not pattern or pattern in f]

    return ret
