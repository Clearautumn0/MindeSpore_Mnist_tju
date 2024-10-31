from src.config import DATA_PATH, LEARNING_RATE, EPOCHS
from src.train import train






if __name__ == '__main__':
    train(data_path=DATA_PATH, learning_rate=LEARNING_RATE, epochs=EPOCHS)