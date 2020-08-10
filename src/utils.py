import numpy as np
import config
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def train_test_split(X, y, test_size=0.3, train_size=None, random_state=None):

    if random_state is not None:
        np.random.seed(seed=random_state)
    if train_size:
        test_size = 1.0 - train_size
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    indx = np.random.choice(size, int(size * test_size))
    return X[~indx], X[indx], y[~indx], y[indx]


def accuracy_score(data1, data2):
    return np.mean(np.array(data1) == np.array(data2))


def save(model):
    if not os.path.exists("../saved_model/"):
        os.makedirs('../saved_model/')
    model.save(config.MODEL_PATH)


def load():
    model = load_model(config.MODEL_PATH)
    return model


def plot_graphs(history):

    for string in ["accuracy", "loss"]:
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()


def save_tokenizer(tokenizer):
    with open(config.TOKENIZER_PATH, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer():
    with open(config.TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
