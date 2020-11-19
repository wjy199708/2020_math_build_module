import sys
from glob import glob
import torch
from PIL import Image
import numpy as np


def eval_one_epoch(model, criteria, test_loaders, batch_size, device):
    model.to(device)
    model.eval()
    
    val_loss = 0
    num_samples = 0
    for i, data in enumerate(test_loaders):
        darkchannelimg, label = data
        label = torch.reshape(label, (-1, 1))
        """ 将测试的数据放入指定的device """
        inputs = darkchannelimg.to(device)
        labels = label.to(device)   

        predict,feature = model(inputs)
        loss = criteria(predict, labels)
        num_samples += labels.shape[0]

        loss = val_loss * batch_size / num_samples
        acc = '{}%'.format(loss*100)

    return loss, acc



def eval(model,dark_channel_img,index=None,fix_model_pth='./out/log/weights_120_v3.pth',label_dir='../data/label/2217_frame_label.txt'):
    from sklearn.preprocessing import MinMaxScaler

    fixed_model=fix_model_pth
    model.load_state_dict(torch.load(fixed_model))

    img=Image.open(dark_channel_img)
    img=np.array(img)
    # img=img.reshape(1,1280,-1)
    img=np.expand_dims(img,axis=0)
    img=np.expand_dims(img,axis=0)

    label_dir=label_dir
    label_list=[]
    with open(label_dir,mode='r') as f:
        label_list=[x for x in f.readlines()]
    
    label_list=np.array(label_list).reshape(-1,1)
    mm=MinMaxScaler()
    label_list=mm.fit_transform(label_list)


    # target=label_list[index-1]
    predict,feature=model(torch.tensor(img,dtype=torch.float32))
    predict=mm.inverse_transform(predict.detach().numpy())
    # print('predict:',predict)
    return predict

