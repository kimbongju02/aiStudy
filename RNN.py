

import numpy as np

class RNN:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros(W_x), np.zeros(W_h), np.zeros(b)]
        
        self.temp = None
        
    def forward(self, input_data, h_prev):
        W_x, W_h, bias = self.params
        t = np.matmul(h_prev, W_h) + np.matmul(input_data, W_x) + bias
        h_next = np.tanh(t)
        
        self.temp = (input_data, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        W_x, W_h, bias = self.params
        input_data, h_prev, h_next = self.temp
        
        dt = dh_next * (1 - h_next**2)
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)  
        dh_prev = np.matmul(dt, W_h.T)
        dWx = np.matmul(input_data.T, dt)
        dx = np.matmul(dt, W_x.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev
    
class TimeRNN:
    def __init__(self, W_x, W_h, b, stateful=False):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros(W_x), np.zeros(W_h), np.zeros(b)]
        self.layers = None
        self.hidden_state = None
        self.dh = None
        self.stateful = stateful
        
    def set_state(self, hidden_state):
        self.hidden_state = hidden_state
        
    def reset_state(self):
        self.hidden_state = None
        
    def forward(self, input_data):
        W_x, W_h, b = self.params
        N, T, D = input_data.shape
        D, H = W_x.shape
        
        self.layers = []
        output = np.empty((N, H), dtype='f')
        
        if not self.stateful or self.hidden_state in None:
            self.hidden_state = np.zeros((N, H), dtype='f')
            
        for t in range(T):
            layer = RNN(*self.params)
            self.hidden_state = layer.forward(input_data[:, t, :], self.h)
            output[:, t, :] = self.hidden_state
            self.layers.append(layer)
            
        return output
    
    def backward(self, doutput):
        W_x, W_h, b = self.params
        N, T, H = doutput.shape
        D, H = W_x.shape
        
        dinput = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(doutput[:, t, :] + dh)
            dinput[:, t, :] = dx
            
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
                
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh
        
        return dinput
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.temp = None
        
    def forward(self, x, h_prev, c_prev):
        W_x, W_h, b = self.params
        N, H = h_prev.shape
        
        A = np.dot(x, W_x) + np.dot(h_prev, W_h) + b
        
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        
        self.temp = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        W_x, W_h, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.temp
        
        tanh_c_next = np.tanh(c_next)
        
        ds = dc_next + (dh_next * c_prev) * (1 - tanh_c_next**2)
        dc_prev = ds * f
        
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * id
        
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2)
        
        dA = np.hstack((df, dg, di, do))
        
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dA, W_x.T)
        dh_prev = np.dot(dA, W_h.T)
        
        return dx, dh_prev, dc_prev