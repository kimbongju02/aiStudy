'''

입력층, 은닉층, 출력층으로 구성
입력층의 노드 크기는 784
은닉층은 4개의 층
은닉층 1층의 노드 크기는 100
은닉층 2층의 노드 크기는 100
은닉층 3층의 노드 크기는 100
은닉층 4층의 노드 크기는 100
출력층의 노드 크기는 10

입력층은 tensorflow.keras.datasets.mnist로 평탄화하여 크기가 (60000, 784)이기에 노드 크기가 784
은닉층은 이유 없다.
출력층은 실제 값이 one-hot encoding으로 나오기에 노드의 크기가 10

최적화 방법으로 Adam 사용
활성화함수는 ReLU, Sigmoid 함수 선택 가능
결정화함수는 softmax 사용

배치정규화 사용
L2 규제 0.15
Dropout 0.5


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

# https://www.notion.so/8f6be99da01840fc891baec700a0c011?pvs=4
import numpy as np
import tensorflow as tf
from collections import OrderedDict

np.random.seed(42)

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10

# 데이터 전처리
x_train = X_train[:10000]
x_test = X_test[:3000]

y_train = y_train[:10000]
y_test = y_test[:3000]

'''
print(x_train.shape)  # (10000, 28, 28)
print(x_test.shape)   # (3000, 28, 28)
print(y_train.shape)  # (10000, )
print(y_test.shpae)   # (3000, )
'''

    # 평탄화
x_train, x_test = x_train.reshape(-1, 28*28).astype(np.float32), x_test.reshape(-1, 28*28).astype(np.float32)

x_train = x_train /.255
x_test = x_test /.255

y_train = np.eye(num_classes)[y_train] # one-hot encoding

'''
print(x_train.shape)  # (10000, 784)
print(x_test.shape)   # (3000, 784)
print(y_train.shape)  # (10000, 10)
print(y_test.shpae)   # (3000, )
'''

# Hyper Parameter
epochs = 1000
lr = 1e-3
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

# Util Functions
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x) # overflow 방지
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(pred_y, true_y):
    if pred_y.ndim ==1:
        true_y = true_y.reshape(1, true_y.size)
        pred_y = pred_y.reshape(1, pred_y.size)
        
    if true_y.size == pred_y.size:
        true_y = true_y.argmax(axis=1)
        
    batch_size = pred_y.shape[0]
    return -np.sum(np.log(pred_y[np.arange(batch_size), true_y])) / batch_size

# SGD
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
# Adam
class Adam:
	def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.iter = 0
		self.m = None
		self.v = None

	def update(self, params, grads):
		if self.m is None:
			self.m, self.v = {}, {}
			for key, val in params.items():
				self.m[key] = np.zeros_like(val)
				self.v[key] = np.zeros_like(val)

		self.iter += 1
		lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
		
		for key in params.keys():
			self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
			self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

			params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key] + 1e-7))

# ReLU
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, input_data):
        self.mask = (input_data <= 0)
        out = input_data.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
# Sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, input_data):
        out = 1 / (1 + np.exp(-input_data))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.dout
        return dx
    
# Layer
class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.input_data = None
        self.input_data_shape = None
        
        self.dW = None
        self.db = None
        
    def forward(self, input_data):
        self.input_data_shape = input_data.shape
        
        input_data = input_data.reshape(input_data.shape[0], -1)
        self.input_data = input_data
        out = np.dot(self.input_data, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.input_data.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.input_data_shape)
        return dx
    
# Batch Normalization
class BatchNormalization():
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape =None

		self.running_mean = running_mean
		self.running_var = running_var

		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None

	def forward(self, input_data, is_train=True):
		self.input_shape = input_data.shape
		if input_data.ndim != 2:
			N, C, H, W = input_data.shape
			input_data = input_data.reshape(N, -1)

		out = self.__forward(input_data, is_train)
	
		return out.reshape(*self.input_shape)

	def __forward(self, input_data, is_train):
		if self.running_mean is None:
			N, D = input_data.shape
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)

		if is_train:
			mu = input_data.mean(axis=0)
			xc = input_data - mu
			var = np.mean(xc**2, axis=0)
			std = np.sqrt(var + 10e-7)
			xn= xc / std
   
			self.batch_size = input_data.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
		else:
			xc = input_data - self.running_mean
			xn = xc / ((np.sqrt(self.running_var + 10e-7)))
		
		out = self.gamma * xn + self.beta
		return out

	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
		
		dx = self.__backward(dout)
  
		dx = dx.reshape(*self.input_shape)
		return dx
	
	def __backward(self, dout):
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(self.xc* dout, axis=0)
		dxn = self.gamma * dout
		dxc = dxn / self.std
		dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
		dvar = 0.5 * dstd / self.std
		dxc += (2.0 / self.batch_size) * self.xc * dvar
		dmu = np.sum(dxc, axis=0)
		dx = dxc - dmu / self.batch_size

		self.dgamma = dgamma
		self.dbeta = dbeta
  
		return dx
    
# Dropout
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
    
# softmax
class Softmax:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, input_data, t):
        self.t = t
        self.y = softmax(input_data)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        if self.t.size == self.y.size: # 정답 레이블의 형태가 one-hot encoding 형태
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx
    
# Model
class MyModel:
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', decay_lamda=0, use_dropout=False,
                 dropout_ratio=0.5, use_batchnorm=False):
        self.inpu_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.decay_lamda = decay_lamda
        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm
        self.params = {}
        
        self.__init_weight(activation)
        
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Layer' + str(idx)] = Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
            
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            
            self.layers['Activation-function' + str(idx)] = activation_layer[activation]()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)
                
        idx = self.hidden_layer_num + 1
        self.layers['Layer' + str(idx)] = Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = Softmax()
        
    def __init_weight(self, activation):
        all_size_list = [self.inpu_size] + self.hidden_size_list + [self.output_size]
        
        for idx in range(1, len(all_size_list)):
            scale = None
            if activation.lower() == 'relu':
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif activation.lower() == 'sigmoid':
                scale = np.sqrt(1.0 / all_size_list[idx-1])
                
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
            
    def predict(self, x, is_train=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, is_train)
            else:
                x = layer.forward(x)  
        return x
    
    def loss(self, x, t, is_train=False):
        y = self.predict(x, is_train)
        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.decay_lamda * np.sum(W**2) # L2 규제
            
        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x, is_train=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t, is_train=True)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Layer' + str(idx)].dW + self.decay_lamda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Layer' + str(idx)].db
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
                
        return grads
    
# model 생성
decay_lambda = 0.15
model = MyModel(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10,
                  decay_lamda=decay_lambda, use_dropout=True, dropout_ratio=0.5, use_batchnorm=True)

optimizer = Adam(lr=lr)

model_train_loss_list = []
model_train_acc_list = [] 
model_test_acc_list = []

for epoch in range(epochs):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    grads = model.gradient(x_batch, y_batch)
    optimizer.update(model.params, grads)
    
    loss = model.loss(x_batch, y_batch)
    model_train_loss_list.append(loss)
    
    train_acc = model.accuracy(x_train, y_train)
    test_acc = model.accuracy(x_test, y_test)
    model_train_acc_list.append(train_acc)
    model_test_acc_list.append(test_acc)
    
    if epoch % 50 == 0:
        print("model Epoch: {} Train Loss : {:.4f} Train Accuracy: {:.4f} Test Accuracy: {:.4f}".format(epoch+1, loss, train_acc, test_acc))