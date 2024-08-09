# from PIL import Image, ImageFilter
# import numpy as np
# from numpy.random import normal
#
# # 读取图像
# image = Image.open('/home/jq/Real/MGLD-VSR-main/dataset/VideoLQ/Input/006/00000000.png')
#
# # 将图像转换为numpy数组
# img_array = np.array(image)
#
# # 生成噪声
# noise = normal(loc=0, scale=0.1, size=img_array.shape)
#
# # 将噪声添加到图像数组中
# noisy_img_array = img_array + noise.astype(img_array.dtype)
#
# # 确保噪声在正确的范围内
# noisy_img_array = np.clip(noisy_img_array, 0, 255)
#
# # 将数组转换回PIL图像
# noisy_image = Image.fromarray(noisy_img_array.astype('uint8'))
#
# # 保存噪声图像
# noisy_image.save('noisy_image.jpg')
#
# # 显示图像
# noisy_image.show()


import cv2
import random
import numpy as np

img = cv2.imread('/home/jq/Real/MGLD-VSR-main/dataset/VideoLQ/Input/006/00000000.png')

# 产生高斯随机数
noise = np.random.normal(90, 100, size=img.size).reshape(img.shape[0], img.shape[1], img.shape[2])

# 加上噪声
img = img + noise
img = np.clip(img, 0, 255)
noise_img = noise
noise_img = np.clip(noise, 0, 255)
# img = img / 255
save_path = 'nosied_image.png'
save_path1 = 'noise.png'
cv2.imwrite(save_path, img)
cv2.imwrite(save_path1, noise_img)
# cv2.imshow('Gauss noise',img)
# cv2.waitKey(0)
