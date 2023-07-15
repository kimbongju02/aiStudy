


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

def filtered_image(image, filter, output_size):
    filtered_image = np.zeros((output_size, output_size))
    filter_size = filter.shape[0]
    
    for i in range(output_size):
        for j in range(output_size):
                # 합성곱 연산
            multiply_values = image[i:(i + filter_size), j:(j + filter_size)] * filter # 필터 연산
            sum_value = np.sum(multiply_values) # 필터 후 폴링
            
            if sum_value > 255:
                sum_value = 255
                
            filtered_image[i, j] = sum_value
            
    return filtered_image


img_url = "https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png"

image = url_to_image(img_url, gray=True)
print("image_shape:", image.shape)

vertical_filter = np.array([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]])
horizontal_filter = np.array([[1., 0., -1.],
                            [2., 0., -2.],
                            [1., 0., -1.]])

output_size = int((image.shape[0] - 3) / 1 + 1)
print("output_size: ", output_size)

vertical_filtered = filtered_image(image, vertical_filter, output_size)
horizontal_filtered = filtered_image(image, horizontal_filter, output_size)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("vertical")
plt.imshow(vertical_filtered, cmap='gray')


plt.subplot(1, 2, 2)
plt.title("horizontal")
plt.imshow(horizontal_filtered, cmap='gray')

plt.show()