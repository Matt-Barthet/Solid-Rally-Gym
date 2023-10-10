import numpy as np


class OnlineScaler:
    def __init__(self):
        self.min = 100000000  # Used if I want to do MinMax later
        self.max = -100000000  # Used if I want to do MinMax later

        self.n = 0  # Used if I want to do Z normalization later
        self.mean = 0  # Used if I want to do Z normalization later
        self.M2 = 0  # Used if I want to do Z normalization later

    def update(self, value):

        self.min = min(self.min, value)
        self.max = max(self.max, value)

        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            return None
        else:
            self.var = self.M2 / (self.n - 1)
            self.std = np.sqrt(self.var)
            return self

    def transform(self, value):
        if self.max == self.min:
            return value  # Avoid division by zero
        else:
            return (value - self.min) / (self.max - self.min)


class VectorScaler:
    def __init__(self, size):
        self.scalers = [OnlineScaler() for _ in range(size)]

    def update(self, vector):
        for i, value in enumerate(vector):
            self.scalers[i].update(value)

    def transform(self, vector):
        return np.array([[self.scalers[i].transform(value) for i, value in enumerate(vector[j])] for j in range(len(vector))])

