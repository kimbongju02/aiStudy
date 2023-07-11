import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')

#하이퍼 파라미터
epochs = 1000   # 반복 횟수
lr = 0.1        # 학습률

#유틸 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(pred_y, true_y): # 평균제곱오차
    return 0.5 * (np.sum((true_y - pred_y)**2))

def cross_entropy_error(pred_y, true_y): # 
    if true_y.ndim == 1:
        true_y = true_y.reshape(1, -1)
        pred_y = pred_y.reshape(1, -1)
        
    delta = 1e-7
    return -np.sum(true_y * np.log(pred_y + delta))

def cross_entropy_error_for_batch(pred_y, true_y):
    if true_y.ndim == 1:
        true_y = true_y.reshape(1, -1)
        pred_y = pred_y.reshape(1, -1)
        
    delta = 1e-7
    batch_size = pred_y.shape[0]
    return -np.sum(true_y * np.log(pred_y + delta)) / batch_size

def cross_entropy_error_for_bin(pred_y, true_y):
    return 0.5 * np.sum((-true_y * np.log(pred_y) - (1 -true_y) * np.log(1 -pred_y)))

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def differential(f, x):
    eps = 1e-5
    diff_value = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        temp_val = x[i]
        
        x[i] = temp_val + eps
        f_h1 = f(x)
        
        x[i] = temp_val - eps
        f_h2 = f(x)
        
        diff_value[i] = (f_h1 - f_h2) / (2 * eps)
        x[i] = temp_val
        
    return diff_value 
    
# 신경망
class LogicGateNet():
    
    def __init__(self):
        
        def weight_init():
            np.random.seed(1)
            weights = np.random.randn(2)
            bias = np.random.rand(1)
            
            return weights, bias
        
        self.weights, self.bias = weight_init()
    
    #예측 결과
    def predict(self, x):
        W = self.weights.reshape(-1, 1)
        b = self.bias
        
        pred_y = sigmoid(np.dot(x, W) + b)
        return pred_y
    
    #손실
    def loss(self, x, true_y):
        pred_y = self.predict(x)
        return cross_entropy_error_for_bin(pred_y, true_y)
    
    def get_gradient(self, x, t):
        
        def loss_grad(grad):
            return self.loss(x, t)
        
        grad_W = differential(loss_grad, self.weights)
        grad_B = differential(loss_grad, self.bias)
        
        return grad_W, grad_B
        
#모델 생성 및 학습
def ANDGate():
    
    AND = LogicGateNet()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [0], [0], [1]])

    train_loss_list = []

    for i in range(epochs):
        grad_W, grad_B = AND.get_gradient(X, Y)
        
        AND.weights -= lr * grad_W
        AND.bias -= lr * grad_B
        
        loss = AND.loss(X, Y) 
        train_loss_list.append(loss)
        
        if i % 100 == 99:
            print("Epoch: {}, Cost: {}, Weights: {}, Bias: {}".format(i+1, loss, AND.weights, AND.bias))
            
    print(AND.predict(X))
    
def ORGate():
    OR = LogicGateNet()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [1]])

    train_loss_list = []

    for i in range(epochs):
        grad_W, grad_B = OR.get_gradient(X, Y)
        
        OR.weights -= lr * grad_W
        OR.bias -= lr * grad_B
        
        loss = OR.loss(X, Y) 
        train_loss_list.append(loss)
        
        if i % 100 == 99:
            print("Epoch: {}, Cost: {}, Weights: {}, Bias: {}".format(i+1, loss, OR.weights, OR.bias))
            
    print(OR.predict(X))
    
def NANDGate():
    NAND = LogicGateNet()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1], [1], [1], [0]])

    train_loss_list = []

    for i in range(epochs):
        grad_W, grad_B = NAND.get_gradient(X, Y)
        
        NAND.weights -= lr * grad_W
        NAND.bias -= lr * grad_B
        
        loss = NAND.loss(X, Y) 
        train_loss_list.append(loss)
        
        if i % 100 == 99:
            print("Epoch: {}, Cost: {}, Weights: {}, Bias: {}".format(i+1, loss, NAND.weights, NAND.bias))
            
    print(NAND.predict(X))
    
class XORNet():
    
    def __init__(self):
        np.random.seed(1)
        
        def weight_init():
            params = {}
            params['w_1'] =  np.random.randn(2)
            params['b_1'] =  np.random.rand(2)
            params['w_2'] =  np.random.randn(2)
            params['b_2'] =  np.random.rand(1)
            return params
        
        self.params = weight_init()
        
    def predict(self, x):
        W_1, W_2 = self.params['w_1'].reshape(-1 ,1), self.params['w_2'].reshape(-1, 1)
        B_1, B_2 = self.params['b_1'], self.params['b_2']
        
        A1 = np.dot(x, W_1) + B_1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W_2) + B_2
        pred_y = sigmoid(A2)
        
        return pred_y
        
    def loss(self, x, true_y):
        pred_y = self.predict(x)
        return cross_entropy_error_for_bin(pred_y, true_y)
    
    def get_gradient(self, x, t):
        def loss_grad(grad):
            return self.loss(x, t)
        
        grads = {}
        grads['w_1'] = differential(loss_grad, self.params['w_1'])
        grads['b_1'] = differential(loss_grad, self.params['b_1'])
        grads['w_2'] = differential(loss_grad, self.params['w_2'])
        grads['b_2'] = differential(loss_grad, self.params['b_2'])
        
        return grads
    
def XORGate():
    XOR = XORNet()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    
    train_loss_list = list()
    
    for i in range(epochs):
        grads = XOR.get_gradient(X, Y)
        
        for key in ('w_1', 'b_1', 'w_2', 'b_2'):
            XOR.params[key] -= lr * grads[key]
        
        loss = XOR.loss(X, Y) 
        train_loss_list.append(loss)
        
        if i % 100 == 99:
            print("Epoch: {}, Cost: {}".format(i+1, loss))
            
    print(XOR.predict(X))

XORGate()