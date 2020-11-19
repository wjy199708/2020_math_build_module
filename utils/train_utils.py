import torch
import tqdm
import time
from torchvision import transforms
from utils._eval import eval_one_epoch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" 定义训练函数 """
def train_model(model, criterion, optimizer, dataload, num_epochs, device, scheduler=None, testloader=None):
    with open('./out/loss/batch_loss.txt',mode='w') as bloss:
        with open('./out/loss/epoch_loss.txt',mode='w') as eloss:
            with tqdm.trange(0, num_epochs, desc='epochs') as tbar,\
                tqdm.tqdm(desc='train', total=len(dataload), leave=False) as pbar:
                    for epoch in tbar:
                        dt_size = len(dataload.dataset)
                        show_dict = {}
                        epoch_loss = 0
                        step = 0
                        if scheduler is not None:
                            scheduler.step(epoch)
                            print('cur_lr:', scheduler.get_lr())

                        for i,data in enumerate(dataload):
                            step += 1
                            darkchannelimg,label=data
                            label=torch.reshape(label,(-1,1))
            
                            inputs = darkchannelimg.to(device)
                            labels = label.to(device)
                            # zero the parameter gradients
                            """ 每一步在对loss进行反向传播计算梯度时，需要清除上一步残留的梯度信息，防止梯度累加 """
                            optimizer.zero_grad()
                            # forward
                            outputs,features = model(inputs)
                            """ 单个训练batch_size损失计算 """
                            # print(type(outputs),type(labels),outputs)
                            """ 加载criteria的损失设计模块 """
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            """ 周期内的损失累加计算 """
                            epoch_loss += loss.item()

                            print(loss,file=bloss)

                            """ loss compute """
                            show_dict.update(
                                {'train_loss': loss.item(), 'epoch_loss': epoch_loss})
                            # print("%d/%d,train_loss:%0.3f" %
                            #       (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
                            pbar.update()
                            pbar.set_postfix(show_dict)
                            tbar.set_postfix(show_dict)
                            tbar.refresh()

                        """ 每一个轮次进行模型评估 """
                        # val_loss,acc=eval_one_epoch(model=model,criteria=criterion,\
                        #     test_loaders=testloader,batch_size=1,device=torch.device('cuda:1'))
                        # print('epoch:{},val_loss:{},acc:{}'.format(epoch,val_loss,acc),file=eloss)
                        # print('='*90,file=eloss)
                        # print("epoch %d loss:%0.3f" % (epoch, epoch_loss))

                        print('epoch:{},epoch_loss:{}'.format(epoch,epoch_loss),file=eloss)
                        print('='*90,file=eloss)

                        pbar.close()
                        pbar = tqdm.tqdm(desc='train', total=len(dataload), leave=False)
                        pbar.set_postfix(dict(step=step))

                        if epoch%5 is 0:
                            torch.save(model.cuda().state_dict(), 'out/log2/weights_{}_v3.pth'.format(epoch))   
                    return model


def get_criteria(prediction,labels):
    if prediction.size()[1] is not 4:
        raise ValueError("input image size is {} not (B,4) ".format(prediction.size()))
    loss_reg_1=nn.SmoothL1Loss()
    loss_reg_2=Log_Cosh()
    a,b,c=1/2,1/4,1/4
    """ prediction划分 """

    loss_total=

""" 定义log余弦回归损失 """
class Log_Cosh(nn.Module):
    def __init__(self):
        super(Log_Cosh,self).__init__()
    def forward(self,pred,target):
        loss = np.log(np.cosh(pred - true))
        return np.sum(loss)

