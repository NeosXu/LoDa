from enum import Enum, auto


class DataloaderMode(Enum):
    train = auto()
    val = auto()
    test = auto()
    inference = auto()
