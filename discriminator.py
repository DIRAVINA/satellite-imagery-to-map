import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, concatenate
from keras.initializers import RandomNormal
from generator import Generator

class Discriminator:
    def __init__(self):
        self.generator = Generator()
        self.initializer = RandomNormal(stddev=0.02, seed=42)

    def build_discriminator(self):
        image = Input(shape=(256, 256, 3), name="ImageInput")
        target = Input(shape=(256, 256, 3), name="TargetInput")
        
        x = concatenate([image, target])
        x = self.generator.downscale(64)(x)
        x = self.generator.downscale(128)(x)
        x = self.generator.downscale(512)(x)

        x = Conv2D(512, kernel_size=4, strides=1, kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(1, kernel_size=4, kernel_initializer=self.initializer)(x)
        
        return Model(inputs=[image, target], outputs=x, name="Discriminator")
