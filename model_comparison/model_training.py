from keras.utils import Sequence
import numpy as np
import os
from skimage.io import imread
from sklearn.utils import shuffle

from albumentations import (
    Blur, Compose, HorizontalFlip, HueSaturationValue,
    IAAEmboss,
    IAASharpen, JpegCompression, OneOf,
    RandomBrightness, RandomBrightnessContrast,
    RandomContrast, RandomCrop, RandomGamma,
    RandomRotate90, RGBShift, ShiftScaleRotate,
    Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion
)
from albumentations import Resize


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir=r'../data/val_test', image_folder='img/', mask_folder='masks/',
                 batch_size=1, image_size=768, nb_y_features=1,
                 augmentation=None,
                 suffle=True):

        self.root_dir = root_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.image_filenames = os.listdir(os.path.join(root_dir, image_folder))
        self.mask_names = os.listdir(os.path.join(root_dir, mask_folder))

        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.suffle = suffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.suffle == True:
            self.image_filenames, self.mask_names = shuffle(
                self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name):
        return (imread(self.root_dir + "/" + self.image_folder + "/" + image_name)/255).astype("float32"), (imread(
            self.root_dir + "/" + self.mask_folder + "/" + mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        """
        Generate one batch of data

        """
        # Generate indexes of the batch
        data_index_min = int(index*self.batch_size)
        data_index_max = int(
            min((index+1)*self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        # The last batch can be smaller than the others
        this_batch_size = len(indexes)

        # Defining dataset
        X = np.empty((this_batch_size, self.image_size,
                     self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size,
                     self.image_size, self.nb_y_features), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):

            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            # if augmentation is defined, we assume its a train set
            if self.augmentation is not None:

                # Augmentation code
                augmented = self.augmentation(self.image_size)(
                    image=X_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(
                    self.image_size, self.image_size, self.nb_y_features)
                X[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm

            # if augmentation isnt defined, we assume its a test set.
            # Because test images can have different sizes we resize it to be divisable by 32
            elif self.augmentation is None and self.batch_size == 1:
                X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i],
                                                          self.mask_names[index * 1 + i])
                augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(
                    X_sample.shape[1]//32)*32)(image=X_sample, mask=y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32),\
                    y_sample.reshape(
                        1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

        return X, y


def aug_with_crop(image_size=256, crop_prob=1):
    return Compose([
        RandomCrop(width=image_size, height=image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04,
                         rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        # Emboss(p=0.25),
        Blur(p=0.01, blur_limit=3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * \
                             0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)
    ], p=1)
