import config
import dataset
import numpy as np
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_data(data, train_data=False):
    if train_data:
        tokenizer = config.TOKENIZER
        tokenizer.fit_on_texts(data)
        utils.save_tokenizer(tokenizer)
    else:
        tokenizer = utils.load_tokenizer()

    seq = tokenizer.texts_to_sequences(data)
    padded_sequence = pad_sequences(
        seq, maxlen=config.MAX_LEN, padding=config.PADDING, truncating=config.TRUNCATING)

    return padded_sequence


def train(model):
    X_train, X_test, y_train, y_test = dataset.get_data()
    X_train = preprocess_data(X_train, train_data=True)
    X_test = preprocess_data(X_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(y_train[:5])

    history = model.train(X_train, y_train, X_test, y_test)
    return history


def predict(data):

    data = preprocess_data(data)
    model = utils.load()
    result = model.predict_classes(data)
    return result[0][0]
