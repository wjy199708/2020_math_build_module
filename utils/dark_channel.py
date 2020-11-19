import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import tqdm
import os
from PIL import Image
from skimage.morphology import disk  #生成扁平的盘状结构元素，其主要参数是生成圆盘的半径
import skimage.filters.rank as sfr
import scipy.misc
import matplotlib

matplotlib.use("Agg")

def zmMinFilterGray(src, r=7):
    '''''最小值滤波，r是滤波器半径'''
    return cv2.erode(src,np.ones((2*r-1,2*r-1)))
# =============================================================================
#     if r <= 0:
#         return src
#     h, w = src.shape[:2]
#     I = src
#     res = np.minimum(I  , I[[0]+range(h-1)  , :])
#     res = np.minimum(res, I[range(1,h)+[h-1], :])
#     I = res
#     res = np.minimum(I  , I[:, [0]+range(w-1)])
#     res = np.minimum(res, I[:, range(1,w)+[w-1]])
# =============================================================================
 #   return zmMinFilterGray(res, r-1)

def guidedfilter(I, p, r, eps):
    '''''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r,r))
    m_p = cv2.boxFilter(p, -1, (r,r))
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip = m_Ip-m_I*m_p

    m_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I = m_II-m_I*m_I

    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I

    m_a = cv2.boxFilter(a, -1, (r,r))
    m_b = cv2.boxFilter(b, -1, (r,r))
    return m_a*I+m_b

def getV1(m, r, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
    '''''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m,2)                                         #得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1,7), r, eps)     #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                              #计算大气光照A
    d = np.cumsum(ht[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax]<=0.999:
            break
    A  = np.mean(m,2)[V1>=ht[1][lmax]].max()

    V1 = np.minimum(V1*w, maxV1)                   #对值范围进行限制

    return V1,A

def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1,A = getV1(m, r, eps, w, maxV1)               #得到遮罩图像和大气光照
    for k in range(3):
        Y[:,:,k] = (m[:,:,k]-V1)/(1-V1/A)           #颜色校正
    Y =  np.clip(Y, 0, 1)
    if bGamma:
        Y = Y**(np.log(0.5)/np.log(Y.mean()))       #gamma校正,默认不进行该操作
    return Y

def getEveryVideoCaptureImgDarkChannel(base_dir):
    count=1
    with tqdm.trange(len(os.listdir('../data/airport/capture'))) as tbar:
        disp_dict={}
        for name in glob('{}/*.jpg'.format(base_dir)):
            count+=1
            # print(name.split('\\')[1])
            m = deHaze(cv2.imread(name)/255.0)*255
            scipy.misc.imsave('../data/airport/defog/defog_{}.jpg'.format(name.split('/')[4].split('.')[0]), m)


            disp_dict.update({'num':count})
            tbar.update()
            tbar.set_description(desc='num')
            tbar.set_postfix(disp_dict)
            tbar.refresh()

# def get_fog_mask(raw_img_dir,dark_channel_img_dir):
#     raw_img=[]
#     dark_channel_img=[]
#     for raw_img_path in glob('{}/*'.format(raw_img_dir)):
#         raw_img.append(raw_img_path)
#     for dark_channel_path in glob('{}/*'.format(dark_channel_img_dir)):
#         dark_channel_img.append(dark_channel_path)
    
#     with tqdm.trange(len(raw_img)) as tbar:
#         for i in tbar:
#             img1=np.array(Image.open(raw_img[i]))
#             img2=np.array(Image.open(dark_channel_img[i]))
#             img3=Image.fromarray(img1-img2)
#             img3.save('../data/fog_img/fog_img_{}.png'.format(i+1))

#             tbar.refresh()

def min_box(image,kernel_size=15):
    min_image = sfr.minimum(image,disk(kernel_size))  #skimage.filters.rank.minimum()返回图像的局部最小值

    return min_image

def calculate_dark(image):
    if not isinstance(image,np.ndarray):
        raise ValueError("input image is not numpy type")  #手动抛出异常
    dark = np.minimum(image[:,:,0],image[:,:,1],image[:,:,2]).astype(np.float32) #取三个通道的最小值来获取暗通道
    dark = min_box(dark,kernel_size=15)
    return dark/255


def get_dark_channel_img(base_dir):
    with tqdm.trange(len(os.listdir(base_dir))) as tbar:
        count=1
        disp_dict={}
        for name in glob('{}/*'.format(base_dir)):
            count=count+1
            # print(name.split('\\')[1])
            haze=np.array(Image.open(name))[:,:,0:3]/255    
            dark = calculate_dark(haze)
            scipy.misc.imsave('../data/airport/dark_channel/darkchannel_{}.jpg'.format(name.split('/')[4].split('.')[0]),dark)


            disp_dict.update({'num':count})
            tbar.update()
            tbar.set_description(desc='num')
            tbar.set_postfix(disp_dict)
            tbar.refresh()



if __name__ == '__main__':
    # m = deHaze(cv2.imread('../data/video_capture/frame_000008.jpg')/255.0)*255
    # cv2.imwrite('defog2.jpg', m)
    # img1=np.array(Image.open('../data/video_capture/frame_000005.jpg'))
    # img2=np.array(Image.open('../data/defog/defog_frame_000005.jpg'))
    # img3=Image.fromarray(img1-img2)
    # img3.save('./res.png')
    # get_fog_mask('../data/video_capture','../data/defog')


    # print('generate dark channel')
    # get_dark_channel_img(base_dir='../data/airport/capture')
    # print('='*90)
    print('defog-ing.....')
    getEveryVideoCaptureImgDarkChannel(base_dir='../data/airport/capture')

    # ''' A大气光照强度的shape '''
    # V1,A=getV1(cv2.imread('../data/video_capture/frame_000010.jpg')/255.0,r=81, eps=0.001, w=0.95, maxV1=0.80)
    # print(A)


