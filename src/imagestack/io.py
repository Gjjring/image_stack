import pickle
import image

def load_image_file(fpath):
    with open(fpath, 'rb') as fp:
        unpickler = pickle.Unpickler(fp)
        image = unpickler.load()
    return image

def save_image_to_file(image, fpath):
    with open(fpath, 'wb') as fp:
        pickler = pickle.Pickler(fp)
        pickler.dump(self)
