import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
import torchvision.transforms as transforms
import torch.nn as nn
import scipy.misc
import torch.utils.data as data
import PIL.Image as Image
import os

from dataset.Dataset import TrainData,TestDataset
from utils.train_utils import train_model
from model.DenseNetCLS import DenseNet
from glob import glob
from utils._eval import eval as eval
import time

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(batch_size):
    model = DenseNet()

    model.to(device)

    model.train()

    """ 数据路径配置 """
    base_dir=os.getcwd()
    """ 暗通道图 """
    root_dir=os.path.join(base_dir,'data','airport')
    label_dir=os.path.join(base_dir,'data','label')

    criterion = nn.SmoothL1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    """ 设置周期迭代更新lr """
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=5, last_epoch=-1)
    traindataset = TrainData(root_dir=root_dir,label_dir=label_dir)
    dataloaders = DataLoader(   
        traindataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testdataset=TestDataset(root_dir=root_dir,label_dir=label_dir)
    testloader=DataLoader(testdataset,batch_size=batch_size)

    train_model(model, criterion, optimizer,
                dataloaders, 100 , device, scheduler=None, testloader=testloader)




""" 测试阶段 """
save_dir = "output"  # 预测图像的存储路径
# 显示模型的输出结果

# 显示模型的输出结果


def test():
    model = DenseNet()

    model.to(device)
    model.load_state_dict(torch.load('./out/log/weights_120_v3.pth'))

    """ 数据路径配置 """
    base_dir=os.getcwd()
    """ 暗通道图 """
    root_dir=os.path.join(base_dir,'data','airport')
    label_dir=os.path.join(base_dir,'data','label')
    test_dataset = TestDataset(root_dir=root_dir,label_dir=label_dir)
    dataloaders = DataLoader(test_dataset, batch_size=4)

    data_len=len(dataloaders.dataset)

    model.eval()
    
    """ 计算预测损失 """
    criteria=nn.SmoothL1Loss()

    with torch.no_grad():

        # img_pth='darkchannel_frame_000911.jpg'
        # img=Image.open('./data/airport/dark_channel/{}'.format(img_pth))
        # img=np.array(img)
        # img=np.expand_dims(img,axis=0)
        # img=np.expand_dims(img,axis=0)
        # img=torch.tensor(img,dtype=torch.float32).to(device)
        # out,feature=model(img)
        # print(out)
        # print(de_normalization(out))
        total_loss=0
        with open('./out/acc/predict_loss.txt',mode='w') as predictLoss:
            for i,data in enumerate(dataloaders):
                darkchannelimg,label=data
                darkchannelimg=darkchannelimg.to(device)
                label=torch.reshape(label,(-1,1))
                label=label.to(device)
                
                predict,feature=model(darkchannelimg)
                loss=criteria(predict,label)

                total_loss+=loss
                print('total_loss:{}'.format(total_loss).numpy(),file=predictLoss,flush=True)
                print('='*90,file=predictLoss,flush=True)
            acc=1-(total_loss/len(dataloaders))
            print('acc:{}'.format(acc),file=predictLoss,flush=True)

def predict_highway():
    with open('./out/highway/eval.txt',mode='w') as f:
        for name in glob('./data/highway/dark_channel_highway/*') :
            predict=eval(DenseNet(),name,fix_model_pth='./out/log2/weights_95_v3.pth'\
                ,label_dir='./data/label/2217_frame_label.txt')
            print('predict:{}'.format(predict),file=f,flush=True)


            
from sklearn.preprocessing import MinMaxScaler
from dataset.Dataset import TrainData
def de_normalization(data):
    data=data.cpu()

    mm=MinMaxScaler()
    train=TrainData('./data/airport',label_dir='./data/label')
    
    label_data=np.array(train.label_data).reshape(-1,1)

    train_label=mm.fit_transform(label_data)

    return mm.transform(data),train.label_data[910]


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str,
                       help="train or test", required=True)
    parse.add_argument("--batch_size", type=int,
                       default=4)  # 此处根据自己电脑的性能选取batch

    # parse.add_argument('--dark_channel_img',type=str,help='dark_channel_img path')
    # parse.add_argument('--index',type=int,help='index')

    args = parse.parse_args()

    if args.action == "train":
        train(args.batch_size)
    elif args.action == "test":
        test()
    elif args.action=='eval':
        predict_highway()

