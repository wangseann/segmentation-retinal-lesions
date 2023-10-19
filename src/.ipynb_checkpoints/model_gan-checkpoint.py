from tensorflow.keras import backend as K
from tensorflow.keras import losses  # <-- Replace 'objectives' with 'losses'
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  
from model_unet import get_unet
from utils.losses import gen_dice_multilabel, dice_coef
import tensorflow as tf  # Import TensorFlow
import numpy as np


def compile_unet(patch_height, patch_width, channels, n_classes, weights):
    """
    It creates, compiles and loads the best weights of the previously trained U-Net
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :param weights: weights of the pre-trained U-Net model
    :return: the compiled U-Net
    """
    unet = get_unet(patch_height, patch_width, channels, n_classes)

    unet.compile(optimizer=Adam(learning_rate=1e-4), loss=gen_dice_multilabel, metrics=['accuracy', dice_coef])
    # unet.compile(optimizer=Adam(learning_rate=1e-4), loss=gen_dice_multilabel, metrics=['accuracy', dice_coef])

    # load the weights of the already trained U-Net model
    unet.load_weights(weights)

    return unet


def get_discriminator(patch_height, patch_width, channels, n_classes):
    """
    It creates the discriminator, compiles and return the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the discriminator
    """

    k = 3  # kernel size
    s = 2  # stride
    n_filters = 32  # number of filters

    inputs = Input((patch_height, patch_width, channels + n_classes))
    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(inputs)
    conv1 = BatchNormalization(scale=True, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=True, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(pool1)
    conv2 = BatchNormalization(scale=True, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=True, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=True, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=True, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=True, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=True, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=True, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=True, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    gap = GlobalAveragePooling2D()(conv5)
    outputs = Dense(1, activation='sigmoid')(gap)

    d = Model(inputs, outputs)

    # loss of the discriminator. it is a binary loss
    def d_loss(y_true, y_pred):
        # Replace 'objectives' with 'losses' in the following lines
        #L = losses.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        L = losses.mean_squared_error(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L
    
    d.compile(optimizer=Adam(learning_rate=1e-4), loss=d_loss, metrics=['accuracy', dice_coef])

    return d


def get_gan(g, d, patch_height, patch_width, channels, n_classes):
    """
    GAN (that binds generator and discriminator)
    It gets the combined and compiles model (U-Net + discriminator)
    :param g: segmentation model
    :param d: discriminative model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the combined model (U-Net + discriminator)
    """

    image = Input((patch_height, patch_width, channels))
    labels = Input((patch_height, patch_width, n_classes))

    fake_labels = g(image)
    fake_pair = Concatenate(axis=3)([image, fake_labels])

    gan = Model([image, labels], d(fake_pair))

    # loss of the combined model. it has a component that penalizes that the discriminator classify the outputs of the
    # unet as fake and another component that penalizes the difference between real and predicted segmentation maps
    def gan_loss(y_true, y_pred):

        #trade-off coefficient
        alpha_recip = 0.05

        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

         # Replace 'objectives' with 'losses' in the following lines
        #L_adv = losses.binary_crossentropy(y_true_flat, y_pred_flat)
        L_adv = losses.mean_squared_error(y_true_flat, y_pred_flat)

        L_seg = gen_dice_multilabel(labels, fake_labels)

        return alpha_recip * L_adv + L_seg

    gan.compile(optimizer=Adam(learning_rate=1e-4), loss=gan_loss, metrics=['accuracy', dice_coef])

    return gan