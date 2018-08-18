from annoy import AnnoyIndex
import pickle
import os

from pdb import set_trace

if __name__ == "__main__":
    file_to_embedding = {}
    for i, f in enumerate(os.listdir('output')):
        # if i >= 3:
        #     break
        with open(os.path.join('output',f), "rb") as handle:
            to_append = pickle.load(handle)
            file_to_embedding = dict(file_to_embedding, **to_append)
    

    num_to_file = {}
    for i, f in enumerate(file_to_embedding.keys()):
        num_to_file[i] = f

    ind = AnnoyIndex(25088)
    for i, f in num_to_file.items():
        if i % 1000 == 0:
            print(i)
        ind.add_item(i, file_to_embedding[f])

    ind.build(10)
    ind.save('test.ann')