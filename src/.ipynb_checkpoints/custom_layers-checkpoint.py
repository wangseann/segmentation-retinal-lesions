# custom_layers.py
import tensorflow as tf

class CustomTrainingLayer(tf.keras.layers.Layer):
    def __init__(self, gan, **kwargs):
        super(CustomTrainingLayer, self).__init__(**kwargs)
        self.gan = gan

    def call(self, img_gan_batch, lab_gan_batch):
        loss, acc, dice = self.gan.train_on_batch(img_gan_batch, lab_gan_batch)
        return loss, acc, dice
