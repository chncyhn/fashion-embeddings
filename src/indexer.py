import pickle
import os
import logging
from pdb import set_trace

from annoy import AnnoyIndex
from tqdm import tqdm

from src.encoder import Encoder

VECTOR_DIM = 4096


class Inferer(Encoder):
    def __init__(self, index_path, num_to_file_path):
        Encoder.__init__(self)
        
        self.ind = AnnoyIndex(VECTOR_DIM)
        self.ind.load(index_path)
        
        with open(num_to_file_path, "rb") as handle:
            self.num_to_file = pickle.load(handle) 

    def get_nns_of_img(self, img_path, num_nn=10):
        '''
        Get nearest neighbours of the image.
        Returns a list of strings (file paths of neighbours).
        '''
        img_encoding = self.encode_image(img_path)
        nn = self.ind.get_nns_by_vector(img_encoding, num_nn)
        nn = map(lambda x: self.num_to_file[x], nn)
        return list(nn)


def build_index(ntrees=10, index_path='test.ann', num_to_file_path='map.pickle'):

    logging.info('Reading encodings.')
    file_to_embedding = {}
    for i, f in enumerate(os.listdir('output')):
        with open(os.path.join('output', f), 'rb') as handle:
            to_append = pickle.load(handle)
            file_to_embedding = dict(file_to_embedding, **to_append)

    # Create integer keys for each file 
    # (for annoy's requirement)
    num_to_file = {}
    for i, f in enumerate(file_to_embedding.keys()):
        num_to_file[i] = f
    
    # Add items to index 
    ind = AnnoyIndex(VECTOR_DIM) 
    for i, f in tqdm(num_to_file.items()):
        ind.add_item(i, file_to_embedding[f])
    
    logging.info('Building index.')
    # Build & save index
    ind.build(ntrees)
    ind.save(index_path)

    with open(num_to_file_path, "wb") as handle:
        pickle.dump(num_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
