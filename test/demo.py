
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk  #生成扁平的盘状结构元素，其主要参数是生成圆盘的半径
import skimage.filters.rank as sfr

def min_box(image,kernel_size=15):
    min_image = sfr.minimum(image,disk(kernel_size))  #skimage.filters.rank.minimum()返回图像的局部最小值

    return min_image

def calculate_dark(image):
    if not isinstance(image,np.ndarray):
        raise ValueError("input image is not numpy type")  #手动抛出异常
    dark = np.minimum(image[:,:,0],image[:,:,1],image[:,:,2]).astype(np.float32) #取三个通道的最小值来获取暗通道
    dark = min_box(dark,kernel_size=15)
    return dark/255

haze = np.array(Image.open("../data/video_capture/frame_000008.jpg"))[:,:,0:3]/255
dark_haze = calculate_dark(haze)

import scipy.misc
scipy.misc.imsave('outfile.jpg', dark_haze)




# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(haze)

# plt.subplot(2,1,2)
# plt.imshow(dark_haze,cmap="gray")

# plt.show()

