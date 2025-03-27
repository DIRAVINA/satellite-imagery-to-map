import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, LeakyReLU, ReLU, concatenate
from keras.initializers import RandomNormal

class Generator:
    def __init__(self):
        self.initializer = RandomNormal(stddev=0.02, seed=42)

    def downscale(self, filters):
        return tf.keras.Sequential([
            Conv2D(filters, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False),
            LeakyReLU(alpha=0.2),
            BatchNormalization()
        ])

    def upscale(self, filters):
        return tf.keras.Sequential([
            Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            ReLU()
        ])

    def build_generator(self):
        inputs = Input(shape=(256, 256, 3))

        encoder = [self.downscale(f) for f in [64, 128, 256, 512, 512, 512, 512]]
        latent_space = self.downscale(512)
        decoder = [self.upscale(f) for f in [512, 512, 512, 512, 256, 128, 64]]
        
        x = inputs
        skips = []
        for layer in encoder:
            x = layer(x)
            skips.append(x)
        
        x = latent_space(x)
        skips = reversed(skips)

        for up, skip in zip(decoder, skips):
            x = up(x)
            x = concatenate([x, skip])

        outputs = Conv2DTranspose(3, kernel_size=4, strides=2, kernel_initializer=self.initializer, activation='tanh', padding='same')(x)
        
        return Model(inputs, outputs, name="Generator")
