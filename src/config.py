from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

LOSS = "binary_crossentropy"
OPTIMIZER = "adam"

BATCH_SIZE = 32
EPOCHS = 100

MODEL_PATH = "../saved_model/Model.h5"
TOKENIZER_PATH = "../saved_model/tokenizer.pickle"
DATA_PATH = "../data/sarcasm.json"
DATA_PATH_V2 = "../data/sarcasm_v2.json"

VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 16
OOV_TOKEN = "<oov>"

TOKENIZER = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)

PADDING = "post"
TRUNCATING = "post"
