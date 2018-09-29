import pickle
import os

from tqdm import tqdm
from keras.applications import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from src.utility import list_files


class Encoder:
    def __init__(self):
        self.model = VGG16(include_top=True, weights='imagenet')
        self.model.layers.pop()
        self.model.layers.pop()
        self.model.outputs = [self.model.layers[-1].output]
        self.model.layers[-1].outbound_nodes = []


    def encode_image(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)

        encoding = self.model.predict(img).flatten()
        return encoding

    def encode_batch(self, img_folder_path, output_path, recursive=True):
        """
        Encode all images in `img_folder_path`.
        Output to `output_path` as a pickled object.
        """
        files = list_files(img_folder_path, pattern='.jpg', recursive=recursive)
        
        file_to_embedding = {}
        cnt = 0
        for i, f in tqdm(enumerate(files)):
            file_to_embedding[f] = self.encode_image(f)

            if i % 1000 == 0:
                print(f"Dumping embed{cnt}.pickle")
                with open(os.path.join(output_path, f"embed{cnt}.pickle"), "wb") as handle:
                    pickle.dump(file_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
                file_to_embedding = {}
                cnt += 1

        return file_to_embedding


if __name__ == "__main__":
    from pdb import set_trace
    enc = Encoder()
    enc_dict = enc.encode_batch('img', 'output', True)