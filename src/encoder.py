from keras.applications import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm
import pickle

from utility import list_files


class Encoder:
    def __init__(self):
        self.model = VGG16(include_top=False, weights='imagenet')

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
        for f in tqdm(files):
            file_to_embedding[f] = self.encode_image(f)

        with open(output_path, "wb") as handle:
            pickle.dump(file_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return file_to_embedding


if __name__ == "__main__":
    from pdb import set_trace
    #enc = Encoder()
    #enc_dict = enc.encode_batch('img', 'output/embed0.pickle', True)
    set_trace()