import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

label2int = {
    "male": 1,
    "female": 0
}
age_label2int = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6,
    "eighties": 6,
    "nineties": 6
}


def load_data(csv_name, is_age, vector_length=193):
    """A function to load gender recognition dataset from `data` folder
    After the second run, this will load from results/features.npy and results/labels.npy files
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if is_age:
        if os.path.isfile("results/features_age.npy") and os.path.isfile("results/labels_age.npy"):
            X = np.load("results/features_age.npy")
            y = np.load("results/labels_age.npy")
            return X, y
    else:
        if os.path.isfile("results/features_gender.npy") and os.path.isfile("results/labels_gender.npy"):
            X = np.load("results/features_gender.npy")
            y = np.load("results/labels_gender.npy")
            return X, y
    # read dataframe
    df = pd.read_csv(csv_name)
    # get total samples
    n_samples = len(df)

    if is_age:
        X = np.zeros((n_samples, vector_length))  # initialize an empty array for all audio features
        y = np.zeros((n_samples, 1))  # initialize an empty array for all audio labels (age)
        for i, (filename, age) in tqdm.tqdm(enumerate(zip(df['filename'], df['age'])), "Loading data", total=n_samples):
            if age in age_label2int:
                features = np.load("data/" + filename)
                X[i] = features
                y[i] = age_label2int[age]

        # save the audio features and labels into files,
        # so we won't load each one of them next run
        np.save("results/features_age", X)
        np.save("results/labels_age", y)
    else:
        # initialize an empty array for all audio features
        X = np.zeros((n_samples, vector_length))
        # initialize an empty array for all audio labels (1 for male and 0 for female)
        y = np.zeros((n_samples, 1))
        for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
            features = np.load("data/"+filename)
            X[i] = features
            y[i] = label2int[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
        np.save("results/features_gender", X)
        np.save("results/labels_gender", y)
    return X, y


def compute_weight(full_label_dict):
    weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.ravel(full_label_dict, order='C')),
                                         y=np.ravel(full_label_dict, order='C'))

    return dict(zip(np.unique(full_label_dict), weights))


def split_data(X, y, test_size=0.1, valid_size=0.1):
    # split training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    # split training set and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    # return a dictionary of values
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }


def create_model_age(vector_length=56):
    """5 hidden dense layers from 256 units to 64."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation="Softmax"))
    model.compile(loss="SparseCategoricalCrossentropy", metrics=["accuracy"], optimizer="adam")

    # print summary of the model
    model.summary()
    return model


def create_model_gender(vector_length=56):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""
    input_shape = (vector_length,)
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),

    ])
    model.add(Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    # print summary of the model
    model.summary()
    return model