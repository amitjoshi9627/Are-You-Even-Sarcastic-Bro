import engine
import utils
from model import SarcasmDetector


def train():
    model = SarcasmDetector()
    history = engine.train(model)
    utils.plot_graphs(history)


if __name__ == "__main__":
    train()
