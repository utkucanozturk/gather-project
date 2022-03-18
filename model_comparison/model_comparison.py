import pandas as pd
from sklearn.model_selection import KFold
from numpy import std
from numpy import mean
from albumentations import Resize
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from tensorflow.keras.utils import Sequence
import numpy as np
from keras.models import load_model
from efficientnet.tfkeras import EfficientNetB4
import cv2




class DataGeneratorFolder(Sequence):
    # mainly based on https://github.com/Diyago/ML-DL-scripts/tree/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline
    def __init__(self, root_dir=r'../data/val_test', image_folder='img/', mask_folder='masks/',
                 batch_size=1, image_size=256, nb_y_features=1,
                 augmentation=None,
                 shuffle=True):

        self.root_dir = root_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.image_filenames = sorted(os.listdir(
            os.path.join(root_dir, image_folder)))
        self.mask_names = sorted(os.listdir(
            os.path.join(root_dir, mask_folder)))

        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.shuffle = shuffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            self.image_filenames, self.mask_names = shuffle(
                self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name, verbose=False):
        if verbose:
            print(f"in read_image_mask. image name: {image_name}")
            print(f"mask name: {mask_name}")
            print("complete path: " + self.root_dir + "/" +
                  self.image_folder + "/" + image_name)
        X_sample, y_sample = (imread(self.root_dir + "/" + self.image_folder + "/" + image_name)/255).astype("float32"), (imread(
            self.root_dir + "/" + self.mask_folder + "/" + mask_name, as_gray=True) > 0).astype(np.int8)
        if np.all(y_sample == 1):
            # some mask formats need special thresholds:
            if verbose:
                print("converting mask format... ")
            X_sample, y_sample = (imread(self.root_dir + "/" + self.image_folder + "/" + image_name)/255).astype("float32"), (imread(
                self.root_dir + "/" + self.mask_folder + "/" + mask_name, as_gray=True) > 0.50).astype(np.int8)
        return X_sample, y_sample

    def get_image_name_at_index(self, index):
        return self.image_filenames[index]

    def get_full_image_name_at_index(self, index):
        return self.root_dir + "/" + self.image_folder + "/" + self.image_filenames[index]

    def __getitem__(self, index, verbose=False):
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
                                                      self.mask_names[index * self.batch_size + i], verbose)

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
                                                          self.mask_names[index * 1 + i], verbose)
                augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(
                    X_sample.shape[1]//32)*32)(image=X_sample, mask=y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32),\
                    y_sample.reshape(
                        1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

        return X, y


def iou_metric(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(),
                           bins=([0, 0.5, 1], [0, 0.5, 1]))

    TN = temp1[0][0][0]
    FP = temp1[0][0][1]
    FN = temp1[0][1][0]
    TP = temp1[0][1][1]

    intersection = temp1[0]

    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    iou = intersection / union
    return iou, TN, FP, FN, TP


def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = y_pred_in
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def invert_predictions(prediction):
    return 1-prediction


invert_predictions_vectorized = np.vectorize(invert_predictions)


class All_0_Predictor:
    def predict(self, X_test_image):
        return np.zeros(shape=(1, X_test_image.shape[1], X_test_image.shape[2], 1))


class All_1_Predictor:
    def predict(self, X_test_image):
        return np.ones(shape=(1, X_test_image.shape[1], X_test_image.shape[2], 1))


def invert_predictions(prediction):
    return 1-prediction


invert_predictions_vectorized = np.vectorize(invert_predictions)
# invert_predictions_vectorized(ypred2)


class UnetArchitecture2EncoderWrapper():
    def __init__(self, fold_number, image_size=256, verbose=False):
        self.fold_number = fold_number
        if image_size == 256:
            if verbose:
                print("load 256 encoder model")
            self.model = load_model(
                f"cached_intermediate_data/model_unet_encoder_fold{fold_number}.h5", compile=False)
        if image_size == 512:
            if verbose:
                print("load 512 encoder model")
            self.model = load_model(
                f"cached_intermediate_data/model_unet_encoder_fold_512{fold_number}.h5", compile=False)

        self.image_size = image_size

    def read_image(self, X_test_image, data_loader, image_index):
        path = data_loader.get_full_image_name_at_index(image_index)
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.image_size, self.image_size))
        x = x/255.0
        return x

    def predict(self, X_test_image):
        IMAGE_DIMENSIONS = (512, 512)
        x = self.read_image(X_test_image)
        y_pred = self.model.predict(np.expand_dims(x, axis=0))[0]  # > 0.5)
        y_pred = cv2.resize(y_pred, IMAGE_DIMENSIONS)
        y_pred = y_pred.reshape(
            (1, IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1))
        return y_pred  # invert_predictions_vectorized(ypred2)


def perform_cv(image_folder, model_names, model_getter, data_loader, n_splits=5, verbose=True):
    cv_outer = KFold(n_splits, shuffle=True, random_state=1)
    image_filenames = sorted(os.listdir(image_folder))
    X = range(0, len(image_filenames))
    fold_number = 0
    df_results_per_image = pd.DataFrame(columns=["model", "image_name", "indexes",
                                                 "iou", "fold_number",
                                                 "TN", "FP", "FN", "TP"])

    for train_ix, test_ix in cv_outer.split(X):
        fold_number = fold_number + 1
        if verbose:
            print("-" * 80)
            print(f"evaluating fold number: {fold_number} of {n_splits}")
        # specify the models to analyze:

        for model_name in model_names:
            if verbose:
                print("-" * 80)
                print(f"evaluating model: {model_name}")
            model = model_getter(model_name, fold_number)
            # here hyperparameter optimization might be added (nested cv) to determine the best_model per fold
            best_model = model

            for i in test_ix:
                if verbose:
                    image_position = np.where(test_ix == i)[0][0]
                    if image_position % 25 == 0:
                        print(
                            f"currently evaluating image number {image_position} of {len(test_ix)}... ")
                X_test_image, y_test_image = data_loader.__getitem__(i)
                y_image_pred = model.predict(X_test_image)
                image_dimensions = X_test_image.shape[1], X_test_image.shape[2]
                y_true_in = y_test_image.reshape(image_dimensions)
                y_pred = y_image_pred.reshape(image_dimensions)
                iou, TN, FP, FN, TP = iou_metric(y_true_in, y_pred)
                df_results_per_image.loc[len(df_results_per_image)] = (
                    model_name,
                    data_loader.get_image_name_at_index(i),
                    i,  iou[0][0], fold_number, TN, FP, FN, TP)
                df_results_per_image.to_pickle(
                    f"results/df_results_cv_all_modelsV4TestUnet.pickle")

    if verbose:
        print('cv done')
    return (df_results_per_image)


def mean_jaccard_score(df):
    # works for dataframes, mainly equivalent to iou_metric
    return df.TP.sum()/(df.TP.sum()+df.FN.sum()+df.FP.sum())
