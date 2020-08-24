import config
import tensorflow as tf
import utils
from tensorflow.keras.layers import GlobalAveragePooling1D, Embedding, Dense, Dropout


class SarcasmDetector:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(Embedding(config.VOCAB_SIZE,
                                 config.EMBEDDING_DIM, input_length=config.MAX_LEN))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(
            loss=config.LOSS, optimizer=config.OPTIMIZER, metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(
            X_train, y_train, epochs=config.EPOCHS, validation_data=(X_test, y_test), verbose=2)
        utils.save(self.model)
        return history
