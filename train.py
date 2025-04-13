import time
import tensorflow as tf
import os
from generator import Generator
from discriminator import Discriminator
from losses import Losses
from dataset import DatasetLoader

class TrainPix2Pix:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_loader = DatasetLoader(dataset_path)
        
        # Load data and verify
        try:
            self.images, self.masks = self.dataset_loader.load_data()
            print(f"Loaded {len(self.images)} satellite images and {len(self.masks)} map images.")
            if len(self.images) == 0 or len(self.masks) == 0:
                raise ValueError("No images or masks loaded. Check dataset format and loader.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Create dataset
        self.data = self.dataset_loader.get_dataset(self.images, self.masks)
        
        # Verify dataset with a single sample
        try:
            iterator = iter(self.data)
            sample_image, sample_mask = next(iterator)
            print(f"Sample batch shape - Image: {sample_image.shape}, Mask: {sample_mask.shape}")
            # Estimate batches based on dataset size
            dataset_size = len(self.images)  # Use loaded image count
            batch_size = 1  # Hardcoded from get_dataset
            self.expected_batches = dataset_size // batch_size
            print(f"Expected batches per epoch: {self.expected_batches}")
            self.data = self.data.shuffle(buffer_size=1000)  # Shuffle without repeat
        except Exception as e:
            print(f"Error creating or validating dataset: {e}")
            raise

        self.generator = Generator().build_generator()
        self.discriminator = Discriminator().build_discriminator()
        
        self.losses = Losses()
        self.gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def train_step(self, inputs, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(inputs, training=True)
            real_output = self.discriminator([inputs, target], training=True)
            fake_output = self.discriminator([inputs, generated_output], training=True)
            
            gen_loss, gen_gan_loss, gen_l1_loss = self.losses.generator_loss(fake_output, generated_output, target)
            disc_loss = self.losses.discriminator_loss(real_output, fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def fit(self, epochs=100):
        for epoch in range(epochs):
            start = time.time()
            print(f"Starting Epoch {epoch+1}/{epochs}")
            batch_count = 0
            # Limit to expected number of batches
            for image, mask in self.data.take(self.expected_batches):
                gen_loss, disc_loss = self.train_step(image, mask)
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Batch {batch_count}: Gen Loss = {gen_loss:.4f}, Disc Loss = {disc_loss:.4f}")
            print(f"Epoch {epoch+1} completed with {batch_count} batches in {time.time() - start:.2f} seconds\n")
            if batch_count == 0:
                print("Warning: No batches processed. Dataset may be empty or misconfigured.")
        
            if(epoch + 1) % 10 == 0: 
                os.makedirs("model", exist_ok=True)
                self.generator.save("model/GAN_Generator.keras")
                self.discriminator.save("model/GAN_Discriminator.keras")
        
        # Save final model
        os.makedirs("model", exist_ok=True)
        self.generator.save("model/GAN_Generator.keras")
        self.discriminator.save("model/GAN_Discriminator.keras")

if __name__ == "__main__":
    try:
        epochs = 100
        dataset_path = "maps/train"
        trainer = TrainPix2Pix(dataset_path)
        trainer.fit(epochs)
    except Exception as e:
        print(f"Training failed: {e}")