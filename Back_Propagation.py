'''

입력층, 은닉층, 출력층으로 구성
입력층의 노드 크기는 784
은닉층은 3개의 층
은닉층 1층의 노드 크기는 100
은닉층 2층의 노드 크기는 62
은닉층 3층의 노드 크기는 32
출력층의 노드 크기는 10
입력층은 tensorflow.keras.datasets.mnist로 평탄화하여 크기가 (60000, 784)이기에 노드 크기가 784
은닉층은 이유 없다.
출력층은 실제 값이 one-hot encoding으로 나오기에 노드의 크기가 10
활성화함수는 ReLU, Sigmoid 함수 선택 가능
결정화함수는 softmax 사용

--hyper Parameter--
epochs = 1000
lr = 1e-3
batch_size = 100
train_size = 784

--학습 기반--
tensorflow

--사용라이브러리--
numpy
matplotlib
OrderedDict

--파이썬버전--
3.11.4

--스펙--
cpu - 11th Gen Intel(R) Core(TM) i5-11300H
ram -24.0GB
system - 64비트 운영 체제, x64 기반 프로세서
graphic card - NVIDIAGeForce RTX 3050 Laptop GPU

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# data load
np.random.seed(42)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10

# 데이터 전처리
X_train, X_test = X_train.reshape(-1, 28 * 28).astype(np.float32), X_test.reshape(-1, 28 * 28).astype(np.float32)
X_train /= .255
X_test /= .255

y_train =np.eye(num_classes)[y_train]
'''
print(X_train.shape) # (60000, 784)
print(X_test.shape)  # (10000, 784)
print(y_train.shape) # (60000, 10)
print(y_test.shape)  # (10000,)
'''

# hyper Parameter
epochs = 1000
lr = 1e-3
batch_size = 100
train_size = X_train.shape[0]

# util Functions
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) # 행에서의 최대값
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x) # 오버플로 방지
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(pred_y, true_y):
    return 0.5 * np.sum((pred_y - true_y)**2)

def cross_entropy_error(pred_y , true_y):
    if pred_y.ndim == 1:
        true_y = true_y.reshape(1, true_y.size) # 1행 true_y.size열 배열
        pred_y = pred_y.reshape(1, pred_y.size) # 1행 pred_y.size열 배열
        
    if true_y.size == pred_y.size: # one-hot encoding
        true_y = true_y.argmax(axis=1)
            
    batch_size = pred_y.shape[0]
    # 예측된 값과 실제 값을 log 취하고 sum
    return -np.sum(np.log(pred_y[np.arange(batch_size), true_y] + 1e-7)) / batch_size

def softmax_loss(X, true_y):
    # softmax를 계산하고 cross_entropy_error하여 나온 loss 반환
    pred_y = softmax(X)
    return cross_entropy_error(pred_y, true_y)

# Sigmoid 함수 https://www.notion.so/6bc622c526274d089e712025a87f49bf?pvs=4
class Sigmoid():
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.dout
        return dx
    
# ReLU 함수 https://www.notion.so/6bc622c526274d089e712025a87f49bf?pvs=4
class ReLU():
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[x < 0] = 0 # making 결과를 x<0이면 0으로 변환
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
         
# Layer
class Layer():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.origin_x_shape = None
        
        self.dL_dW = None # 가중치의 미분값에 손실값의 미분값을 나눈 것
        self.dL_db = None # 편향의 미분값에 손실값의 미분값을 나눈 것
        
    def forward(self, x):
        self.origin_x_shape = x.shape
        
        x = x.reshape(x.shape[0], -1) # 행의 길이가 x.shape[0]인 배열
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dL_dW = np.dot(self.x.T, dout)
        self.dL_db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.origin_x_shape) # origin_x_shape에 들어있는 값으로
        return dx
    
# Softmax
class Softmax():
    def __init__(self):
        self.loss = None
        self.y = None # 최종 출력
        self.t = None # one-hot-encoding
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # 최종 은닉층 통과 값을 softmax 처리
        self.loss = cross_entropy_error(self.y, self.t) # softmax 처리된 값과 실제 값의 오차 비교
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        if self.t.size == self.y.size: # 정답 레이블이 one-hot encoding일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx
    
# model
class MyModel():
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num= len(hidden_size_list)
        self.params= {}
        
        self.__init_weights(activation)
        
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Layer' + str(idx)] = Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
            # activation_layer에서 현재 activation 값을 받아온 후 해당하는 딕셔너리 값을 Activation_function 넣는다
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
        idx = self.hidden_layer_num + 1
        self.layers['Layer' + str(idx)]= Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
        
        self.last_layer = Softmax()
        
    def __init_weights(self, activation):
        weight_std = None
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if activation.lower() == 'relu':
                weight_std = np.sqrt(2.0 / self.input_size) # He 초기화
            elif activation.lower() == 'sigmoid':
                weight_std = np.sqrt(1.0 / self.input_size) # Xavier 초기화
                
            self.params['W' + str(idx)] = weight_std * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.random.randn(all_size_list[idx])
            
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x) # Activation_function에 해당하는 함수의 forward
            
        return x
    
    def loss(self, x, true_y):
        pred_y = self.predict(x)
        
        return self.last_layer.forward(pred_y, true_y) # 최종 출력값의 순전파
    
    def accuracy(self, x, true_y):
        pred_y = self.predict(x)
        pred_y = np.argmax(pred_y, axis=1)
        
        if true_y.ndim != 1:
            true_y = np.argmax(true_y, axis=1)
            
        accuracy = np.sum(pred_y == true_y) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.last_layer.backward(dout) # 최종 출력값의 역전파(기울기)
        
        layers =list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Layer' + str(idx)].dL_dW
            grads['b' + str(idx)] = self.layers['Layer' + str(idx)].dL_db
            
        return grads
    
# model 생성 및 학습
model = MyModel(28*28,[100, 64, 32], 10, activation='relu')

train_lost_list = []
train_acc_list = []
test_acc_list =[]

for epoch in range(epochs):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    grad = model.gradient(x_batch, y_batch)
    
    for key in model.params.keys():
        model.params[key] -= lr * grad[key]
        
    loss = model.loss(x_batch, y_batch)
    train_lost_list.append(loss)
    
    if epoch % 50 == 0:
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print("Epoch: {} Train Accuracy: {:.4f} Test Accuracy: {:.4f}".format(epoch+1, train_acc, test_acc))
        