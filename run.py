import numpy as np
from utils import load_affNIST
from model import STModel

if __name__ == "__main__":
    # x, y = load_affNIST()
    model = STModel((40, 40))
    print model.summary()
