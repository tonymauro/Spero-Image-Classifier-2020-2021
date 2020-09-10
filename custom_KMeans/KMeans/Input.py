from PIL import Image
import numpy as np

def get_arr(path):
    return np.array(Image.open(path))