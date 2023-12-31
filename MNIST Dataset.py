'''

입력층, 은닉층, 출력층으로 구성
입력층의 노드는 784개,
은닉층은 1층
1층 은닉층의 노드는 64개
출력층의 노드는 10개
입력층은 mnist의 입력값이 (60000, 28, 28)의 배열이기에 행*열의 크기
은닉층은 이유 없다.
출력층은 mnist의 실제 결과를 바이너리 10자리로 표현
활성화함수는 sigmoid함수 사용
결정화함수는 softmax사용

--hyper Parameter--
epochs = 2
lr = 0.1
batch_size = 100
train_size = 784

--학습 기반--
tensorflow 

--사용라이브러리--
numpy
matplotlib
time
tqdm.notebook

--파이썬버전--
3.11.4

--스펙--
cpu - 11th Gen Intel(R) Core(TM) i5-11300H
ram -24.0GB
system - 64비트 운영 체제, x64 기반 프로세서
graphic card - NVIDIAGeForce RTX 3050 Laptop GPU

'''

# https://www.notion.so/4f93933a748d4bd2a0d429aa69bb136a?pvs=4
# 다중 클래스 분류
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time 
from tqdm.notebook import tqdm

mnist = tf.keras.datasets.mnist #데이터 셋

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(x_test.shape)  # (10000, 28, 28)
print(y_test.shape)  # (10000,)
"""

    # 데이터 전처리
def flatten_for_mnist(x):
    temp = np.zeros((x.shape[0], x[0].size))

    for idx, data in enumerate(x):      # enumerate -> index와 index에 들은 값
        temp[idx, :] = data.flatten()   # x의 각 행을 평탄화하여 temp[idx]에 할당
        
    return temp

    # 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0 # 색은 0~255까지 있는데 구분할때는 색 필요없으니 정규화

x_train = flatten_for_mnist(x_train)
x_test = flatten_for_mnist(x_test)

"""
print(x_train.shape) # (60000, 784)
print(x_test.shape)  # (10000, 784)
"""

y_train_ohe = tf.one_hot(y_train, depth = 10).numpy() # TensorFlow의 이미지의 실제값을 바이너리로 표현 5 -> [100000]
y_test_ohe = tf.one_hot(y_test, depth = 10).numpy()

"""
print(y_train_ohe.shape) # (60000, 10)
print(y_test_ohe.shape)  # (10000, 10)
"""

#하이퍼 파라미터
epochs = 2      # 반목 횟수
lr = 0.1        # 학습률
batch_size = 100
train_size = x_train.shape[0]

#유틸 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Mean_Squared_Error(pred_y, true_y):
  return np.mean(np.sum(np.square((true_y - pred_y))))

def cross_entropy_error(pred_y, true_y):
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

    # https://www.notion.so/ba32a87def8d484da3fcf47d46952c5e?pvs=4
def cross_entropy_error_for_bin(pred_y, true_y):
    return 0.5 * np.sum((-true_y * np.log(pred_y) - (1 - true_y) * np.log(1 -pred_y)))

    # https://www.notion.so/softmax-60159b1b3b6a48558bfe65ab60ef7a4d?pvs=4
def softmax(a): # 개선전 수식
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

    # 기울기   Epsilon-delta argument 사용
    ''' 미분이란 t_1에서 t_0을 뺀 값에서 미분을 시작하는데 t[i]만 사용하는 상황에서 오차를 줄이기 위하여
        가장 작은 값이 eps를 사용하여 미분'''
def differential_1d(f, x): #1차원
    eps = 1e-5  # Epsilon
    diff_value = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        temp_val = x[i]
        
        x[i] = temp_val + eps
        f_h1 = f(x)             # f = loss_grad
        
        x[i] = temp_val - eps
        f_h2 = f(x)
        
        diff_value[i] = (f_h1 - f_h2) / (2 * eps)
        x[i] = temp_val
        
    return diff_value

def differential_2d(f, X): # 2차원
    if X.ndim == 1:
        return differential_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X): # 전체 x만큼 반복
            grad[idx] = differential_1d(f, x)
        
        return grad
    
class MyModel():
    
    def __init__(self):
        
        def weight_init(input_nodes, hidden_nodes, output_units):
            np.random.seed(777)
            
            params = {}
            params['w_1'] =  0.01 * np.random.randn(input_nodes, hidden_nodes) # input_nodes*hidden_nodes인 배열의 0~1배열 생성
            params['b_1'] =  np.zeros(hidden_nodes)
            params['w_2'] =  0.01 * np.random.randn(hidden_nodes, output_units)
            params['b_2'] =  np.zeros(output_units)
            return params
        
        self.params = weight_init(784, 64, 10)
        
    def predict(self, x):   # 입력 x에 대한 에측 값
        W_1, W_2 = self.params['w_1'], self.params['w_2']
        B_1, B_2 = self.params['b_1'], self.params['b_2']
        
        A1 = np.dot(x, W_1) + B_1   # 784행 배열과 784행 63열 배열의 스칼라 연산
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W_2) + B_2
        pred_y = softmax(A2) # 다중신경망이니 마지막에 선택을 위해서
        
        return pred_y
        
    def loss(self, x, true_y):
        pred_y = self.predict(x)
        return cross_entropy_error_for_bin(pred_y, true_y) # 교차 엔트로피 바이너리
    
        # 예측값에서 최대값과 실제값을 비교하여 같은지 확인
    def accuracy(self, x, true_y):
        pred_y = self.predict(x)
        y_argmax = np.argmax(pred_y, axis=1) # argmin 최솟값의 인덱스, axis=1이면 열에 따라서 찾는다
        t_argmax = np.argmax(true_y, axis=1)
        
        accuracy = np.sum(y_argmax == t_argmax) / float(x.shape[0])
        return accuracy
    
    def get_gradient(self, x, t):
        def loss_grad(grad):
            return self.loss(x, t) # 예상 값과 실제 값의 오차 계산
        
        grads = {}
        grads['w_1'] = differential_2d(loss_grad, self.params['w_1'])
        grads['b_1'] = differential_2d(loss_grad, self.params['b_1'])
        grads['w_2'] = differential_2d(loss_grad, self.params['w_2'])
        grads['b_2'] = differential_2d(loss_grad, self.params['b_2'])
        
        return grads
    
    
model = MyModel()

train_loss_list = list()
train_acc_list = list()
test_acc_list = list()
iter_per_epoch = max(train_size / batch_size, 1)

start_time = time.time()
for i in tqdm(range(epochs)): # 반복에대한 진행률    표시 tqdm
    batch_idx = np.random.choice(train_size, batch_size) # 0~train_size(784)-1 범위에서 batch_size(100)만큼 batch_idx 추출
    x_batch = x_train[batch_idx]                         # 입력값
    y_batch = y_train_ohe[batch_idx]                     # 입력 값의 실제 결과 값
    
    grads = model.get_gradient(x_batch, y_batch)
    
    for key in grads.keys():
            #가중치 업데이트
        model.params[key] -= lr * grads[key]
        
    loss = model.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    
    train_accuracy = model.accuracy(x_train, y_train_ohe)
    test_accuracy = model.accuracy(x_test, y_test_ohe)
    train_acc_list.append(train_accuracy)
    test_acc_list.append(test_accuracy)
    
    print("Epoch: {}, Cost: {}, Train Accuracy: {}, Test Accuracy: {}".format(i+1, loss, train_accuracy, test_accuracy))
    
end_time = time.time()

print("학습시간: {:.3f}s".format(end_time - start_time))

