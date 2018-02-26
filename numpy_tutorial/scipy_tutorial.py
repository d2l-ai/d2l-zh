"""
Scipy基于numpy, 并提供了更多的函数运算和工程应用
"""

from scipy.misc import imread, imsave, imresize

img = imread('assets/cat.jpg')
print(img.dtype)
print(img.shape)

img_tinted = img * [1, 0.95, 0.9]
img_tinted = imresize(img_tinted, (300, 300))

print(img_tinted.shape)

imsave('assets/cat_tinted.jpg', img_tinted)


"""
Distance between points
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

d = squareform(pdist(x, 'euclidean'))
print(d)

"""
Matplotlib
"""
import matplotlib.pyplot as plt
# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()
#
# # 绘制多个图形
# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Cosine'])
# plt.show()
#
#
# # subplots
# plt.subplot(2, 1, 1)
#
# # Make the first plot
# plt.plot(x, y_sin)
# plt.title('Sine')
#
# plt.subplot(2, 1, 2)
# plt.plot(x, y_cos)
# plt.title('Cosine')
#
# plt.show()
#
# Images
# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()

print(img_tinted.shape)
print(img.shape)