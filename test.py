import tensorflow as tf  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from collections import OrderedDict

np.random.seed(42)
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
num_classes=10

X_train, X_test = X_train.reshape(-1,28*28).astype(np.float32),X_test.reshape(-1,28*28).astype(np.float32)
X_train/=.255
X_test/= .255

Y_train = np.eye(num_classes)[Y_train]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

epochs=1000
learning_Rate=1e-3
batch_size=100
train_size=X_train.shape[0]

def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x), axis=0)
        return y.T
    x=x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def mean_squared_erre(pred_y, true_y):
    return 0.5*np.sum((pred_y - true_y)**2)

def cross_entropy_errer(pred_y,true_y):
    if pred_y.ndim==1:
        true_y = true_y.reshape(1, true_y.size)
        pred_y = pred_y.reshape(1, pred_y.size)
    
    if true_y.size == pred_y.size:
        true_y = true_y.argmax(axis=1)
    
    batch_size = pred_y.shape[0]
    return -np.sum(np.log(pred_y[np.arrange(batch_size), true_y]+1e-7))/batch_size

def softmax_loss(X,true_y):
    pred_y = softmax(X)
    return cross_entropy_errer(pred_y, true_y)

class ReLU():
    def __init__(self):
        self.out=None
    
    def forward(self, x):
        self.mask = (x<0)
        out = x.copy()
        out[x<0] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
        
class Sigmoid():
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1/ (1 + np.exp(-x))
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.dout
        return dx
    
class Layer():
    def __init(self, W, b):
        self.W=W
        self.b = b
        
        self.x = None
        self.origin_X_shape = None
        
        self.DL_dW = None
        self.dL_db = None
        
    def forward(self,x):
        self.origin_X_shape = x.shape
        
        x= x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x , self.W) +self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dL_dW = np.dot(self.x.T, dout)
        self.dL_db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.origin_X_shape)
        return dx
    
class Softmax():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t =None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_errer(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx =self.y.copu()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx
    
class MyModel():
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}
        
        self.__init_weights(activation)
        
        activation_layer = {'sigmoid':Sigmoid, 'relu':ReLU}
        self.layers =OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Layer' + str(idx)] = Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.laers['Activation_function' + str(idx)] = activation_layer[activation]()
        
        idx = self.hidden_layer_num + 1
        
        self.layers['Layer' + str(idx)] = Layer(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = Softmax()
        
    def __init_weights(self, activation):
        weight_Std = None
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if activation.lower() == 'relu':
                weight_Std = np.sqrt(2.0 / self.input_size)
            elif activation.lower() == 'sigmoid':
                weight_Std = np.sqrt(1.0 / self.input_size)
                
            self.params['W' + str(idx)] =weight_Std * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.random.randn(all_size_list[idx])
            
    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)

    def loss(self, x, true_y):
        pred_y = self.pridict(x)
        
        return self.last_layer.forward(pred_y, true_y)
    
    def accuracy(self, x, true_y):
        pred_y =self.predict(x)
        pred_y = np.argmax(pred_y, axis=1)
        
        if true_y.ndim != 1:
            true_y = np.argmax(true_y, axis=1)
            
        accuracy = np.sum(pred_y == true_y) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout  =1
        dout=self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverss()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' +str(idx)] =self.layers['Layer' + str(idx)].dL_dW
            grads['b' +str(idx)] =self.layers['Layer' + str(idx)].dL_db
            
        return grads
    
    
    
model =MyModel(28*28, [100, 64, 32], 10, activation = 'relu')