import tensorflow as tf

class Losses:
    def __init__(self):
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(self, disc_generated, generated_output, target_image):
        gan_loss = self.adversarial_loss(tf.ones_like(disc_generated), disc_generated)
        l1_loss = tf.reduce_mean(tf.abs(target_image - generated_output))
        total_loss = (100 * l1_loss) + gan_loss
        return total_loss, gan_loss, l1_loss

    def discriminator_loss(self, real_output, generated_output):
        real_loss = self.adversarial_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.adversarial_loss(tf.zeros_like(generated_output), generated_output)
        return real_loss + fake_loss
