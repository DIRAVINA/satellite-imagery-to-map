import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img
from glob import glob

class DatasetLoader:
    def __init__(self, path, num_images=1000):
        self.path = path
        self.num_images = num_images
    
    def load_data(self):
        combined_images = sorted(glob(self.path + "*.jpg"))[:self.num_images]
        images = np.zeros((len(combined_images), 256, 256, 3))
        masks = np.zeros((len(combined_images), 256, 256, 3))

        for idx, img_path in enumerate(combined_images):
            combined_image = tf.cast(img_to_array(load_img(img_path)), tf.float32)
            image = combined_image[:, :600, :]
            mask = combined_image[:, 600:, :]
            images[idx] = tf.image.resize(image, (256, 256)) / 255
            masks[idx] = tf.image.resize(mask, (256, 256)) / 255
        
        return images, masks

    def get_dataset(self, images, masks, batch_size=32):
        return tf.data.Dataset.from_tensor_slices((images, masks)).batch(batch_size, drop_remainder=True)
