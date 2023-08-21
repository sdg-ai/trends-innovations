from abc import ABC, abstractmethod

class TandIClassifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, text):
        pass