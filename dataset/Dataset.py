import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os 
from glob import glob
import numpy as np


""" 1850帧的数据 """
""" 北京时间0时开始取帧 """
""" 选择前1500帧的数据进行训练 """
class TrainData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir


        self.img_list=self.get_img_list()
        self.label_data=self.VIS_Normalization(np.array(self.get_label()[:1500]).astype('float'))
        self.darkchannelimg=self.get_darkchannle_img()

    def __getitem__(self,index):

        # img=Image.open(self.img_list[index])
        # img=np.array(img)
        # # img=transforms.ToTensor(img)

        label=float(self.label_data[index])

        label=torch.tensor(label)
        # label=transforms.Normalize(0.5,0.5)(label)

        darkchannel_img=self.dark_channel_img_process(self.darkchannelimg[index])

        return  darkchannel_img,label

    def get_img_list(self):
        img_list=[]
        for name in glob('{}/capture/*'.format(self.root_dir)):
            # print(name)
            img_list.append(name)
        
        return img_list[:300]
    
    def get_darkchannle_img(self):
        img_list=[]
        for name in glob('{}/dark_channel/*'.format(self.root_dir)):
            # print(name)
            img_list.append(name)
        
        return img_list[:300]

    def get_label(self):
        label_txt=os.path.join(self.label_dir,'2217_frame_label.txt')
        label_list=[]
        with open(label_txt,mode='r') as f:
            label_list=[x for x in f.readlines()]
        return label_list

    def __len__(self):
        
        return len(self.darkchannelimg)
    
    def VIS_Normalization(self,data):
        m = np.mean(data)
        mx = max(data)
        mn = min(data)
        return [(float(i) - m) / (mx - mn) for i in data]

    """ 根据pytorch的数据要求，对获取的暗通道图片进行处理 """
    def dark_channel_img_process(self,data):
        darkchannelimg=Image.open(data)
        darkchannelimg=np.array(darkchannelimg)
        darkchannelimg=torch.unsqueeze(torch.tensor(darkchannelimg,dtype=torch.float32),dim=0)
        return darkchannelimg

""" 根据提供数据和视频的时长，选择截取的1500帧到1850帧作为测试数据，进行每一轮次的评估 """
class TestDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir

        self.label_data=self.VIS_Normalization(np.array(self.get_label()[300:400]).astype('float'))
        self.darkchannelimg=self.get_darkchannle_img()[300:400]

    def __getitem__(self,index):
        label=float(self.label_data[index])

        label=torch.tensor(label)
        # label=transforms.Normalize(0.5,0.5)(label)

        darkchannel_img=self.dark_channel_img_process(self.darkchannelimg[index])
        
        return  darkchannel_img,label

    def get_img_list(self):
        img_list=[]
        for name in glob('{}/capture/*'.format(self.root_dir)):
            # print(name)
            img_list.append(name)
        
        return img_list
    
    def get_darkchannle_img(self):
        img_list=[]
        for name in glob('{}/dark_channel/*'.format(self.root_dir)):
            # print(name)
            img_list.append(name)
        
        return img_list

    def get_label(self):
        label_txt=os.path.join(self.label_dir,'2217_frame_label.txt')
        label_list=[]
        with open(label_txt,mode='r') as f:
            label_list=[x for x in f.readlines()]
        return label_list

    def __len__(self):
        
        return len(self.darkchannelimg)
    
    """ VIS数据归一化，加速网络训练和计算 """
    def VIS_Normalization(self,data):
        m = np.mean(data)
        mx = max(data)
        mn = min(data)
        return [(float(i) - m) / (mx - mn) for i in data]
    
    def dark_channel_img_process(self,data):
        darkchannelimg=Image.open(data)
        darkchannelimg=np.array(darkchannelimg)
        darkchannelimg=torch.unsqueeze(torch.tensor(darkchannelimg,dtype=torch.float32),dim=0)
        return darkchannelimg

    

if __name__ == "__main__":

    data=TrainData('../data/airport',label_dir='../data/label')
    # labels=data.VIS_Normalization(np.array(data.get_label()).astype('float'))
    # print(labels)
    # print(data.get_img_list())
    # trainDataloader=DataLoader(data,batch_size=4,num_workers=2)
    # # for i,data in enumerate(trainDataloader):
    # #     print(data)

    # for i,data in enumerate(trainDataloader):
    #     print(data)
