import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import os
import re
from torchvision import transforms as tfs
import random
from model import *
from torch.autograd import Variable
import torch.nn.functional as F

#超参数
batch=12
epoch=128
net_pths=['net_fpn.pth','net_pred1.pth','net_pred2.pth','net_pred3.pth']
citar_1,citar_2,citar_3=1,1,1   #三项预测每项所占的比例
chan_of_fpn=16  #FPN提取特征时的通道数

def load_data(pth,change=False):#读入数据，change的意思是是否重新写入.npy文件
    name_of_disc=['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12']#椎间盘的名字
    files=os.listdir(pth)
    if ('x_train.npy' in files  and 'y_train.npy' in files) and change==False:
        x=np.load(pth+'/x_train.npy',allow_pickle=True)
        y=np.load(pth+'/y_train.npy',allow_pickle=True)
        return x,y,np.size(x,axis=0)
    else:
        x=[]
        y=[]
        txt_list=glob.glob(pth+'/*.txt')
        for i in txt_list:
            z=i.replace(pth,'')
            z=z.replace('.txt','')
            with open(i,'r')as f:
                img=np.array(Image.open(pth+z+'.jpg'))
                tem_x=list(range(11, 0, -1))
                tem_y=list(range(11, 0, -1))
                key=0
                for j in f.readlines():
                    #随机偏移
                    rad_x=random.randint(-5,5)
                    rad_y=random.randint(-5,5)
                    line=j.strip('\n')
                    if any(word if word in line else False for word in name_of_disc):#如果确认是椎间盘
                        loc=re.findall(r"\d+\.?\d*",line)
                        loc_of_disc=re.findall(r"disc\': \'v\d+",line)
                        num=re.findall(r"\d+",str(loc_of_disc))
                        if len(num):
                            num=num[0]
                        else:
                            continue
                        loc=list(map(int,loc))
                        # tem_y.append(int(num)-1)
                        #竖直方向为160个像素，水平方向为80个像素
                        y1=max(loc[1]-80,0)+rad_y
                        y2=min(loc[1]+80,512)+rad_y
                        x1=max(loc[0]-40,0)+rad_x
                        x2=min(loc[0]+40,512)+rad_x
                        z=img[y1:y2,x1:x2]
                        if np.size(z,0)!=160 or np.size(z,1)!=80:#如果越界了就resize为160*80
                            z=np.resize(z,(160,80))
                        #下面对标签进行识别并记录
                        if name_of_disc[0] in line:
                            tem_x[0] = z
                            tem_y[0] = int(num) - 1
                            key += 1
                        elif name_of_disc[1] in line:
                            tem_x[2] = z
                            tem_y[2] = int(num) - 1
                            key += 1
                        elif name_of_disc[2] in line:
                            tem_x[4] = z
                            tem_y[4] = int(num) - 1
                            key += 1
                        elif name_of_disc[3] in line:
                            tem_x[6] = z
                            tem_y[6] = int(num) - 1
                            key += 1
                        elif name_of_disc[4] in line:
                            tem_x[8] = z
                            tem_y[8] = int(num) - 1
                            key += 1
                        elif name_of_disc[5] in line:
                            tem_x[10] = z
                            tem_y[10] = int(num) - 1
                            key += 1
                    else:#确认为椎骨
                        loc = re.findall(r"\d+\.?\d*", line)
                        loc = list(map(int, loc))
                        #同样为160*80的大小
                        y1 = max(loc[1] - 80, 0) + rad_y
                        y2 = min(loc[1] + 80, 512) + rad_y
                        x1 = max(loc[0] - 40, 0) + rad_x
                        x2 = min(loc[0] + 40, 512) + rad_x
                        z = img[y1:y2, x1:x2]
                        # 如果越界了就热size为160*80
                        if np.size(z,0)!=160 or np.size(z,1)!=80:
                            z=np.resize(z,(160,80))
                        loc_of_disc = re.findall(r"vertebra\': \'v\d+", line)
                        num = re.findall(r"\d+", str(loc_of_disc))
                        if len(num):
                            num=num[0]
                        # 下面对标签进行识别并记录
                        if 'L1' in line:
                            tem_x[1] = z
                            tem_y[1] = int(num) +4
                            key += 1
                        elif 'L2' in line:
                            tem_x[3] = z
                            tem_y[3] = int(num) +4
                            key += 1
                        elif 'L3' in line:
                            tem_x[5] = z
                            tem_y[5] = int(num) +4
                            key += 1
                        elif 'L4' in line:
                            tem_x[7] = z
                            tem_y[7] = int(num) +4
                            key += 1
                        elif 'L5' in line:
                            tem_x[9] = z
                            tem_y[9] = int(num) +4
                            key += 1
                # 如果标签不足则直接抛弃该样本
                if key!=11:
                    continue
                x.append(tem_x)
                y.append(tem_y)
        x=np.array(x)
        y=np.array(y)
        np.save(pth+'/x_train',x,allow_pickle=True, fix_imports=True)
        np.save(pth +'/y_train',y,allow_pickle=True, fix_imports=True)
        return x,y,np.size(x,axis=0)

#训练函数
def train(x,y,nets,Loss,opt,opt_2,data_num,echo):
    with tqdm(total=echo) as tq:#用于显示训练的进程
        all_li=torch.randperm(data_num)
        x_train=x[all_li[:],...]
        y_train=y[all_li[:],...]
        for i in range(echo):
                li = torch.randint(0, high=int(data_num), size=(1,batch)).squeeze()#随机挑选batchsize张图片用作训练
                x_in = x_train[li, ...]
                target = y_train[li, ...].squeeze()
                target=target.view(1,batch*11)
                target=target.squeeze()
                for z in range(5):#每个batch都会被训练5次
                    y1,y2,y3= nets[0](x_in)     #首先经过特征提取网络获得3组特征
                    y_pred=citar_1*nets[1](y1)+citar_2*nets[2](y2)+citar_3*nets[3](y3)#将预测结果加权
                    y_pred = y_pred.view(batch * 11, 7)
                    target=target.long()
                    loss = Loss(y_pred, target) #计算损失
                    opt.zero_grad()
                    opt_2.zero_grad()
                    loss.backward()     #误差反向传递并更新参数
                    opt.step()
                    opt_2.step()
                tq.set_postfix(loss=loss)
                tq.update(1)

        torch.save(nets[0],net_pths[0])
        torch.save(nets[1],net_pths[1])
        torch.save(nets[2],net_pths[2])
        torch.save(nets[3],net_pths[3])
        print('The train is finished')

def dataenhance(x,y,n):#数据增广，作水平翻转和垂直翻转
    h=tfs.RandomHorizontalFlip(1)(x)
    v=tfs.RandomVerticalFlip(1)(x)
    x=torch.cat((x,h),0)
    x=torch.cat((x,v),0)
    z=torch.cat((y,y),0)
    y=torch.cat((y,z),0)
    return x,y,3*n


def functional_conv2d(im):#定义sobel算子卷积操作
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='double')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    for i in range(11):
        if i==0:
            edge_detect = F.conv2d(Variable(im[:,i,:,:].unsqueeze(dim=1)), weight,padding=(1,1))
        else:
            edge_detect=torch.cat((edge_detect,F.conv2d(Variable(im[:,i,:,:].unsqueeze(dim=1)), weight,padding=(1,1))),dim=1)
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect



if __name__=='__main__':
    pth = './脊柱疾病智能诊断/train/data'
    pth_of_model = './'
    files = os.listdir(pth_of_model)
    x, y, num_of_data = load_data(pth)  #载入数据
    x = x.astype(float)
    y = y.astype(int)
    x = torch.from_numpy(x).double()
    y = torch.from_numpy(y).int()
    x = functional_conv2d(x)    #sobel算子提取边缘特征
    x,y,num_of_data=dataenhance(x,y,num_of_data)
    weight = torch.tensor([1, 1, 3, 10,10, 1, 1], dtype=torch.double)#前5类是椎间盘，后2类是椎骨。鉴于椎间盘中类别为v4，v5的数量较少，采用加权交叉熵的损失函数
    continue_to_train = False#决定是否从上次的模型继续训练
    if continue_to_train:
        if all(word if word in files else False for word in net_pths):
            fpn = torch.load(net_pths[0]).eval()
            pred16_1 = torch.load(net_pths[1])
            pred16_2 = torch.load(net_pths[2])
            pred16_3 = torch.load(net_pths[3])
        else:
            print('No old models,here are the new ones')
            fpn = FPN(chan_of_fpn).double().eval()
            pred16_1 = Pred(in_size=[int(chan_of_fpn * 2), 160, 80]).double()
            pred16_2 = Pred(in_size=[int(chan_of_fpn * 1.5), 80, 40]).double()
            pred16_3 = Pred(in_size=[int(chan_of_fpn), 40, 20]).double()
    else:
        fpn = FPN(chan_of_fpn).double().eval()
        pred16_1 = Pred(in_size=[int(chan_of_fpn*2),160, 80]).double()
        pred16_2 = Pred(in_size=[int(chan_of_fpn*1.5),80, 40]).double()
        pred16_3 = Pred(in_size=[int(chan_of_fpn),40, 20]).double()
    if torch.cuda.is_available():#gpu版本
        x=x.cuda()
        y=y.cuda()
        weight=weight.cuda()
        fpn=fpn.cuda()
        pred16_1=pred16_1.cuda()
        pred16_2=pred16_2.cuda()
        pred16_3=pred16_3.cuda()
    nets = [fpn, pred16_1, pred16_2, pred16_3]


    loss = nn.CrossEntropyLoss(weight)
    opt_1 = torch.optim.Adam(fpn.parameters(), lr=5e-5)
    opt_2 =torch.optim.Adam([ {'params': nets[1].parameters()},
                              {'params': nets[2].parameters()},
                              {'params': nets[3].parameters()}]
                              ,lr=5e-5)
    train(x, y, nets, loss, opt_1,opt_2, num_of_data, epoch)