



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


img_url = "https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png"

image = url_to_image(img_url, gray=False)
print("image_shape:", image.shape)  # (440, 440, 3)

plt.imshow(image)

    # (r, g, b)
image_copy = image.copy()
image_copy[:, :, 1] = 0
image_copy[:, :, 2] = 0
image_red = image_copy

image_copy = image.copy()
image_copy[:, :, 0] = 0
image_copy[:, :, 2] = 0
image_green = image_copy

image_copy = image.copy()
image_copy[:, :, 0] = 0
image_copy[:, :, 1] = 0
image_blue = image_copy


fig = plt.figure(figsize=(12, 8))

title_list = ['R', 'G', 'B', 'R-grayscale', 'G-grayscale', 'B-grayscale']
image_list = [image_red, image_green, image_blue, image_red[:, :, 0], image_green[:, :, 1], image_blue[:, :, 2]]

for i, image in enumerate(image_list):
    ax = fig.add_subplot(2, 3, i+1)
    ax.title.set_text("{}".format(title_list[i]))
    
    if i >= 3:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
        
plt.show()