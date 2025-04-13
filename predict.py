import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths
VAL_DIR = 'maps/val'
MODEL_PATH = 'model/GAN_Generator_70.keras'
IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_SAMPLES = 5

def load_val_data(val_dir, img_height, img_width):
    """
    Load and preprocess validation images from the maps/val directory.
    Assumes images are paired (satellite + map) in a single file, split horizontally.
    """
    image_paths = [os.path.join(val_dir, fname) for fname in os.listdir(val_dir) if fname.endswith(('.jpg', '.png'))]
    satellite_images = []
    map_images = []

    for path in image_paths:
        # Load image
        img = load_img(path, target_size=(img_height, img_width * 2))  # Double width for paired images
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]

        # Split into satellite and map
        satellite = img[:, :img_width, :]
        map_img = img[:, img_width:, :]

        satellite_images.append(satellite)
        map_images.append(map_img)

    return np.array(satellite_images), np.array(map_images)

def show_predictions(generator, satellite_images, map_images, num_samples=5):
    """
    Display predictions for num_samples images from the validation set.
    Shows satellite image, ground truth map, and predicted map.
    """
    # Randomly select indices
    indices = np.random.choice(len(satellite_images), num_samples, replace=False)

    for idx in indices:
        # Get satellite image and ground truth map
        satellite = satellite_images[idx]
        ground_truth = map_images[idx]

        # Predict map using generator
        satellite_input = tf.expand_dims(satellite, axis=0)  # Add batch dimension
        predicted = generator.predict(satellite_input, verbose=0)[0]  # Remove batch dimension

        # Ensure compatible data type and range for visualization
        satellite_vis = (satellite * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        ground_truth_vis = (ground_truth * 255).astype(np.uint8)
        predicted_vis = np.clip(predicted, 0, 1)  # Ensure within [0, 1]
        predicted_vis = (predicted_vis * 255).astype(np.uint8)

        # Create a figure with 3 subplots
        plt.figure(figsize=(15, 5))

        # Satellite Image
        plt.subplot(1, 3, 1)
        plt.imshow(satellite_vis)
        plt.title("Satellite Image")
        plt.axis('off')

        # Ground Truth Map
        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth_vis)
        plt.title("Ground Truth Map")
        plt.axis('off')

        # Predicted Map
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_vis)
        plt.title("Predicted Map")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():
    # Load the generator model
    try:
        generator = tf.keras.models.load_model(MODEL_PATH)
        print("Generator model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load validation data
    try:
        satellite_images, map_images = load_val_data(VAL_DIR, IMG_HEIGHT, IMG_WIDTH)
        print(f"Loaded {len(satellite_images)} validation images.")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return

    # Show predictions
    show_predictions(generator, satellite_images, map_images, num_samples=NUM_SAMPLES)

if __name__ == "__main__":
    main()