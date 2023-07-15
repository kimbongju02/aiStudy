


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


img_url = "https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png"

image = url_to_image(img_url, gray=False)
print("image_shape:", image.shape)  # (440, 440, 3)

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(image)

filter1 = np.random.randn(3, 3, 3)

filtered_image1 = conv_op(image, filter1)
print(filtered_image1.shape)

plt.subplot(1, 3, 2)
plt.title("Used Filter")
plt.imshow(filter1, cmap='gray')


plt.subplot(1, 3, 3)
plt.title("Result")
plt.imshow(filtered_image1[0, 0, :, :], cmap='gray')
plt.show()
