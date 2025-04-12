# import numpy as np
# import tensorflow as tf
# from keras.utils import img_to_array, load_img
# from glob import glob

# class DatasetLoader:
#     def __init__(self, path, num_images=1000):
#         self.path = path
#         self.num_images = num_images
    
#     def load_data(self):
#         combined_images = sorted(glob(self.path + "*.jpg"))[:self.num_images]
#         images = np.zeros((len(combined_images), 256, 256, 3))
#         masks = np.zeros((len(combined_images), 256, 256, 3))

#         for idx, img_path in enumerate(combined_images):
#             combined_image = tf.cast(img_to_array(load_img(img_path)), tf.float32)
#             image = combined_image[:, :600, :]
#             mask = combined_image[:, 600:, :]
#             images[idx] = tf.image.resize(image, (256, 256)) / 255
#             masks[idx] = tf.image.resize(mask, (256, 256)) / 255
        
#         return images, masks

#     def get_dataset(self, images, masks, batch_size=32):
#         return tf.data.Dataset.from_tensor_slices((images, masks)).batch(batch_size, drop_remainder=True)

import tensorflow as tf
import os
from glob import glob

class DatasetLoader:
    def __init__(self, path, num_images=None):
        self.path = path
        self.num_images = num_images  # Set to None to use all images

    def load_data(self):
        """Load paired satellite and map images from dataset_path."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset path {self.path} does not exist.")
        
        # Ensure path ends with a separator
        path = self.path if self.path.endswith('/') else self.path + '/'
        image_paths = sorted(glob(path + "*.jpg"))
        
        if not image_paths:
            raise ValueError(f"No .jpg files found in {self.path}. Found {len(os.listdir(self.path))} files.")
        
        if self.num_images is not None:
            image_paths = image_paths[:self.num_images]
        
        images = []
        masks = []
        for idx, img_path in enumerate(image_paths):
            try:
                # Load and decode image
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.cast(img, tf.float32)
                
                # Verify image shape (expecting 512x256 for Pix2Pix paired images)
                if img.shape[1] != 1200 or img.shape[0] != 600:
                    print(f"Skipping {img_path}: Expected 1200x600, got {img.shape[1]}x{img.shape[0]}")
                    continue
                
                # Split into satellite (left) and map (right)
                w = img.shape[1] // 2
                satellite = img[:, :w, :]
                map_img = img[:, w:, :]
                
                images.append(satellite)
                masks.append(map_img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No valid paired images loaded after processing.")
        
        return tf.convert_to_tensor(images), tf.convert_to_tensor(masks)

    def preprocess(self, image, mask):
        """Preprocess images and masks."""
        image = tf.image.resize(image, [256, 256])
        mask = tf.image.resize(mask, [256, 256])
        # Normalize to [-1, 1] as common for Pix2Pix
        image = (image / 127.5) - 1
        mask = (mask / 127.5) - 1
        return image, mask

    def get_dataset(self, images, masks, batch_size=1):
        """Create a tf.data.Dataset from images and masks."""
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset