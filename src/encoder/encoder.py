from keras.applications import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

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


if __name__ == "__main__":
    from pdb import set_trace
    set_trace()
