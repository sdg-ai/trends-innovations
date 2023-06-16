from abc import ABC, abstractmethod

class TandIClassifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, text):
        pass

    @abstractmethod
    def train(self, train_data, val_data):
        pass

    @abstractmethod
    def test(self, test_data):
        pass