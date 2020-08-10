import json
import config
from utils import train_test_split

headlines, labels = [], []

datastore = [json.loads(line) for line in open(config.DATA_PATH, 'r')]

for item in datastore:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])

X_train, X_test, y_train, y_test = train_test_split(
    headlines, labels, test_size=0.3)


def get_data():
    return X_train, X_test, y_train, y_test
