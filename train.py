import time
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from losses import Losses
from dataset import DatasetLoader

class TrainPix2Pix:
    def __init__(self, dataset_path):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.images, self.masks = self.dataset_loader.load_data()
        self.data = self.dataset_loader.get_dataset(self.images, self.masks)

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
            
            gen_loss, _, _ = self.losses.generator_loss(fake_output, generated_output, target)
            disc_loss = self.losses.discriminator_loss(real_output, fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def fit(self, epochs=100):
        for epoch in range(epochs):
            start = time.time()
            print(f"Epoch {epoch+1}/{epochs}")
            for image, mask in self.data:
                self.train_step(image, mask)
            print(f"Epoch {epoch+1} completed in {time.time() - start:.2f} seconds\n")

        self.generator.save("model/GAN_Generator.h5")
        self.discriminator.save("model/GAN_Discriminator.h5")

if __name__ == "__main__":
    trainer = TrainPix2Pix("maps/train")
    trainer.fit(100)
