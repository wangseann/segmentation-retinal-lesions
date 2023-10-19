from __future__ import print_function
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage.io import imread
from skimage.transform import rotate
from tensorflow.keras.utils import to_categorical
from random import shuffle, randint
from data import image_shape, compute_statistics_file
import sys
import matplotlib.image as mpimg
from os.path import join, isfile
from os import listdir
from skimage.io import imread

class UNetGeneratorClass(object):
    def __init__(self, data_path, n_class=5, batch_size=32, channels=3, apply_augmentation=True, thres_score=None, train=True):
        if train:
            path = join(data_path, 'train/')
        else:
            path = join(data_path, 'val/')

        self.labels_path = join(path, 'labels/')
        self.image_path = join(path, 'images/')
        self.path = path

        self.files = [f for f in listdir(self.image_path) if isfile(join(self.image_path, f)) and f.endswith('.png')]

        if not self.files:
            raise ValueError("No image files found in the specified image directory.")

        if thres_score is not None:
            files_aux = []
            for file_name in self.files:
                a, b, c = file_name.split("_")
                label_img = imread(join(self.labels_path, a + '_' + b + '_label_' + c))
                score = compute_statistics_file(label_img)
                if score > thres_score:
                    files_aux.append(file_name)
            self.files = files_aux

        if not self.files:
            raise ValueError("No image files meet the specified thres_score criteria.")

        self.img_shape = image_shape(join(self.image_path, self.files[0]))
        self.channels = channels

        self.batch_size = batch_size
        self.n_class = n_class

        self.apply_augmentation = apply_augmentation

# You will need to define the `image_shape` and `compute_statistics_file` functions or import them from somewhere.



    def data_augmentation(self, data, delta):
        """
        It applies a random trasnformation to the data
        :param data: data to apply the transformation
        :param delta: random value between 0 and 3
        :return: the transformed data
        """

        h_i_mirror = np.fliplr(data)
        v_i_mirror = np.flipud(data)

        if delta == 0:
            return data
        elif delta == 1:
            return h_i_mirror
        elif delta == 2:
            return v_i_mirror
        elif delta == 3:
            return rotate(data, angle=180)

    def generate(self):
        """
        It yields images and labels for every batch to train the model
        """
        while True:

            shuffle(self.files)

            for n_batch in range(len(self.files) // self.batch_size):
                batch_files = self.files[n_batch*self.batch_size:(n_batch+1)*self.batch_size]

                images_batch = np.zeros((self.batch_size, self.img_shape[0], self.img_shape[1], self.channels), dtype=np.uint8)
                labels_batch = np.zeros((self.batch_size, self.img_shape[0], self.img_shape[1], self.n_class), dtype=np.uint8)

                n = 0
                for file_name in batch_files:

                    img = imread(self.image_path + file_name)
                    image_array = np.asarray(img).astype(np.uint8)

                    a, b, c = file_name.split("_")

                    label_name = a + '_' + b + '_label_' + c
                    label_img = mpimg.imread(self.labels_path + label_name)
                    label_array = np.asarray(label_img[:, :, 0]).astype(np.uint8)
                    
                    try:
                        assert (np.amin(label_array) >= 0 and np.amax(label_array) <= 5)
                    except AssertionError:
                        print(f"Error: Label array in file {file_name} contains unexpected values: min={np.amin(label_array)}, max={np.amax(label_array)}")
                        # Optionally log or visualize the erroneous label_array for further inspection.

                    if self.apply_augmentation:
                        delta = randint(0, 3)
                        img_augm = self.data_augmentation(image_array, delta)
                        images_batch[n, :, :, :] = img_augm
                        labels_augm = self.data_augmentation(label_array, delta)
                        labels_batch[n, :, :, :] = to_categorical(labels_augm, self.n_class).reshape(labels_augm.shape + (self.n_class,))
                    else:
                        images_batch[n, :, :, :] = image_array
                        labels_batch[n, :, :, :] = to_categorical(label_array, self.n_class).reshape(label_array.shape + (self.n_class,))
                    n += 1

                yield (images_batch, labels_batch)