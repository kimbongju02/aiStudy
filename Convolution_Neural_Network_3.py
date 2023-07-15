


import cv2
import numpy as np
import urllib
import requests
import matplotlib.pyplot as plt
from io import BytesIO

# Util Function
def url_to_image(url, gray=False):
    resp = urllib.request.urlopen(url) # 이미지 주소 받아오기
    
        # 이미지 주소를 부호 없는 8비트 정수형인 바이트 배열로 변환하고 numpy 배열로 변환한다.
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    
    if gray == True:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) # 이미지를 흑백으로 변환
    else:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) # 이미지를 컬러로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    return image

def conv_op(image, kernel, pad=0, stride=1):
    H, W, C = image.shape
    kernel_size =kernel.shape[0]
    
    out_h = (H + 2*pad - kernel_size) // stride + 1
    out_w = (W + 2*pad - kernel_size) // stride + 1
    
    filtered_img = np.zeros((out_h, out_w))
    
        #  [(pad, pad), [pad, pad], (0, 0)]은 각 차원별로 적용할 패딩의 크기를 지정 
        #  여기서 pad는 각 차원의 양쪽에 추가할 패딩의 크기
        #  'constant'는 패딩을 적용할 때 사용할 값의 종류를 지정
    img = np.pad(image, [(pad, pad), [pad, pad], (0, 0)], 'constant') # 패딩 적용
    
    for i in range(out_h):
        for j in range(out_w):
            for c in range(C):
                multiply_values = image[i:(i + kernel_size), j:(j + kernel_size), c] * kernel_size # 필터 연산
                sum_value = np.sum(multiply_values) # 필터 연산 후 폴링
                
                filtered_img[i, j] += sum_value
                
    filtered_img = filtered_img.reshape(1, out_h, out_w, -1).transpose(0, 3, 1, 2)
    
    return filtered_img.astype(np.uint8)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0): # 이미지의 값을 2차원 배열로 전환
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
        # input_data와 동일한 차원을 갖게 되며, 주어진 pad 값만큼의 추가적인 값으로 채워짐  
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    
        # 샘플 개수 N, 채널 개수 C, 채널의 높이 filter_h, 채널의 너비 filter_w, 필터 크기 out_h * out_w
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride] # 스트라이드 적용
            
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0): # col 값을 이미지로 변환
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
        # 샘플 개수 N, 채널 개수 C, 채널의 높이 filter_h, 채널의 너비 filter_w, 필터 크기 out_h * out_w
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
        # 샘플 개수 N, 채널 개수 C, 채널 높이 H + 2*pad + stride -1, 채널 너비 W + 2*pad + stride -1
    img = np.zeros(N, C, H + 2*pad + stride -1, W + 2*pad + stride -1)
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            
                # img 배열의 특정 부분에 col 배열의 값 더하는 연산
                # img의 특정 부분은 stride로 지정된 간격으로 슬라이싱되며, col의 값은 해당 위치의 값과 더해짐
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            
    return img[:, :, pad:H +pad, pad:W + pad]

class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        self.input_data = None
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None
        
    def forward(self, input_data):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1
        
        col = im2col(input_data, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        
        out = np.dot(col, col_W) + self.b
        output = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.input_data = input_data
        self.col = col
        self.col_W = col_W
        
        return output
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN,C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.input_data.shape, FH, FW, self.stride, self.pad)
        
        return dx
    
def init_weigit(num_filters, data_dim, kernel_size, stride=1, pad=0, weight_std=0.01):
    weights = weight_std * np.random.randn(num_filters, data_dim, kernel_size, kernel_size)
    biases = np.zeros(num_filters)
    
    return weights, biases
    

img_url = "https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png"
image_color = url_to_image(img_url, gray=False)
image_color = np.expand_dims(image_color.transpose(2, 0, 1), axis=0)

print("imaga_gary_shape: ", image_color.shape)

W4, b4 =init_weigit(num_filters=32, data_dim=3, kernel_size=3, stride=3)
conv4 = Conv2D(W4, b4, stride=3)
output4 = conv4.forward(image_color)
print("image_color_Conv_Layer_size ", output4.shape)

plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.title("Filter16")
plt.imshow(output4[0, 15, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Filter21")
plt.imshow(output4[0, 20, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Filter32")
plt.imshow(output4[0, 31, :, :], cmap='gray')

plt.show()