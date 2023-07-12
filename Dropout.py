

import numpy as np

class Dropout():
    def __init__(self, dropdout_ratio=0.5):
        self.dropdout_ratio = dropdout_ratio
        self.mask = None
        
    def forward(self, input_data, is_train=True):
        if is_train:
            self.mask = np.random.rand(*input_data.shape) > self.dropdout_ratio
            return input_data * self.mask
        else:
            return input_data * (1.0 - self.dropdout_ratio)
        
    def backward(self, dout):
        return dout * self.mask