# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:22:17 2021

@author: 53412
"""
# -*- coding: utf-8 -*-
from scipy.odr import Model

"""
Created on Mon Apr 12 21:13:34 2021

@author: 53412
"""
import  numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import glob
from PIL import Image
import os
from skimage import io
warnings.filterwarnings("ignore")
 
data_transform = transforms.Compose(
     [transforms.Resize([30,20]),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
# def load_data():
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     train_dataset = torchvision.datasets.CIFAR10(root='data/',train=True,transform=transform,download=True)
#     train_data=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
#     test_dataset = torchvision.datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)
#     test_data=torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
#
#     return train_data,test_data



class my_dataset(Dataset):
    def __init__(self, store_path, split_T,split_L,label_T,label_L, data_transform=None):
        self.store_path = store_path
        self.split_L = split_L
        self.split_T = split_T
        self.label_T=label_T.flatten()
        self.label_L=label_L.flatten()
        #self.name = name
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        self.cur_num_T=0
        self.cur_num_L=0

        for file in glob.glob(self.store_path + '/' + split_T + '/*.jpg'):
            cur_path = file.replace('\\', '/')
            #cur_label = cur_path.split('_')[-1].split('.png')[0]
            self.img_list.append(cur_path)
            self.label_list.append(self.label_T[self.cur_num_T])
            self.cur_num_T=self.cur_num_T+1
        for file in glob.glob(self.store_path + '/' + split_L + '/*.jpg'):
            cur_path = file.replace('\\', '/')
            #cur_label = cur_path.split('_')[-1].split('.png')[0]
            self.img_list.append(cur_path)
            self.label_list.append(self.label_L[self.cur_num_L])
            self.cur_num_L=self.cur_num_L+1
 
    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        #img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label
 
    def __len__(self):
        return len(self.img_list)
        
        


def define_loss():
    Loss=torch.nn.CrossEntropyLoss()
    return Loss

def define_optimizer():
    learnig_rate=1e-4
    optimizer=torch.optim.Adam(net.parameters(),lr=learnig_rate)
    return optimizer


class basic(nn.Module):
    def __init__(self,in_ch,out_ch,strides,padding,need_de):
        super(basic,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.need_de=need_de
        self.strides=strides
        self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=strides,padding=padding)
        self.b1=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=padding)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.de=nn.Conv2d(self.in_ch,self.out_ch,1,stride=self.strides)
    def forward(self,x):
        y=self.conv1(x)
        y=self.b1(y)
        y=F.relu(y)
        y=self.conv2(y)
        y=self.b2(y)
        if self.need_de:
            x=self.de(x)
            y=y+x
        else:
            y=y+x
        y = F.relu(y)
        return y



class Resnet(nn.Module):
    def __init__(self,blocks,in_size,in_ch):
        super(Resnet,self).__init__()
        self.blocks=blocks
        self.in_size=in_size
        self.filters=[64,128,256,512]
        self.conv1=nn.Conv2d(in_ch,self.filters[0],kernel_size=3,stride=1,padding=1)
        self.b=nn.BatchNorm2d(self.filters[0])
        self.layer1=self.make_layer(blocks[0],self.filters[0],self.filters[0],False)
        self.layer2 = self.make_layer(blocks[1], self.filters[0], self.filters[1], True)
        self.layer3 = self.make_layer(blocks[2], self.filters[1], self.filters[2], True)
        self.layer4 = self.make_layer(blocks[3], self.filters[2], self.filters[3], True)
        self.avgpool=nn.AvgPool2d(1)
        # self.fc=nn.Linear(8*in_size*in_size,10)
        self.fc = nn.Linear(512*4*3, 7)

    def make_layer(self,blocks,in_ch,out_ch,need_de):
        layers=[]
        for block in range(blocks):
            if need_de and block==0:
                layers.append(basic(in_ch,out_ch,strides=2,padding=1,need_de=True))
            elif block==0:
                layers.append(basic(in_ch, out_ch, strides=1, padding=1, need_de=False))
            else:
                layers.append(basic(out_ch,out_ch,strides=1,padding=1,need_de=False))
        return nn.Sequential(*layers)

    def forward(self,x):
        out=F.relu(self.b(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avgpool(out)
        out=out.view(out.size()[0],out.size()[1]*out.size()[2]*out.size()[3])
        # out=out.view(-1,8*self.in_size*self.in_size)
        out=self.fc(out)
        return out


    


def train(batch,epoch,train_data,net,optimizer,Loss):
    accuracy=[]
    iterator=[]
    accuracy_train=[]
    it=1
    for i in range(epoch):
        length=len(train_data)
        acc_=0.0
        sum_=0.0
        sum_train=0.0
        acc_train=0.0
        with tqdm(total=length,desc="{}/{}".format(i+1,epoch)) as pbar:
            for index,data in enumerate(train_data):
                if index<=length*9/10:
                    inputs,outputs=data
                    inputs,outputs=inputs.cuda(),outputs.cuda()
                    inputs=inputs.permute(0,1,2,3)
                    out_pred=net(inputs)
                    outputs=outputs.long()
                    loss=Loss(out_pred,outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _,pred=torch.max(out_pred,axis=1)
                    acc_train+=torch.sum(pred==outputs).item()
                    sum_train+=outputs.shape[0]
                    pbar.set_postfix({"loss":loss.item()})
                    pbar.update(1)
                else:
                    inputs,outputs=data
                    inputs,outputs=inputs.cuda(),outputs.cuda()
                    inputs=inputs.permute(0,1,2,3)
                    out_pred=net(inputs)
                    _,pred=torch.max(out_pred,axis=1)
                    acc_+=torch.sum(pred==outputs).item()
                    sum_+=outputs.shape[0]
                    pbar.set_postfix({"val_accuracy":"{:.2f}%".format(100*acc_/sum_)})
                    pbar.update(1)
        accuracy.append(100*acc_/sum_)
        accuracy_train.append(100*acc_train/sum_train)
        iterator.append(it)
        it+=1

    net_path="net.pth"
    torch.save(net,net_path)
    return net_path,accuracy,accuracy_train,iterator

def test(test_data,net_path,Loss):
    net = torch.load(net_path)
    acc=0.0
    sum_=0.0
    batch=32
    length=len(test_data)
    with tqdm(total=length,desc="test") as pbar:
        for index,data in enumerate(test_data):
            inputs,outputs=data
            inputs,outputs=inputs.cuda(),outputs.cuda()
            # inputs=inputs.unsqueeze(0)
            out_pred=net(inputs)
            _,pred=torch.max(out_pred,axis=1)
            acc+=torch.sum(pred==outputs).item()
            sum_+=outputs.shape[0]
            pbar.set_postfix({"test_accuracy":"{:.2f}%".format(100*acc/sum_)})
            pbar.update(1)
    print("\ntest_accuray:{:.2f}%".format(100*acc/sum_))



def Resnet_18(predict_data,net_path):
    #(x,y,z)=predict_data.shape
    net=Resnet(blocks=[3,4,6,3],in_size=32,in_ch=3)
    img = Image.fromarray(predict_data).convert('RGB')
    img = img.resize((30,20))
    predict_data = np.array(img)
    predict_data = np.reshape(predict_data,(1,3,30,20))
    if os.path.exists(net_path):
        net=torch.load(net_path)
    in_put=torch.tensor(predict_data)
    in_put=in_put.float()
    in_put= in_put.cuda()
    #in_put = in_put.permute(0, 1, 2, 3)
    out_put=net(in_put)
    out_put=out_put.cpu()
    op=out_put.detach().numpy()
    #result=out_put.item()
    result=np.argmax(op)
    return result 






if __name__=='__main__':
    # store_path='D://1AA学习//深度学习//实验课//traindata'
    # store_path = store_path.replace('\\', '/')
    # split_T = 'T'
    # split_L = 'L'
    # l_pth='D://1AA学习//深度学习//实验课//traindata//L//L_numpy.npy'
    # l_pth=l_pth.replace('\\','/')
    # t_pth='D://1AA学习//深度学习//实验课//traindata//T//T_numpy.npy'
    # t_pth = t_pth.replace('\\', '/')
    # label_T=np.load(t_pth)
    # label_l=np.load(l_pth)
    # label_L=label_l+2
    # name = { 'A1':0, 'A2':1, 'A3':2, 'A4':3, 'A5':4, 'B1':5, 'B2':6}
    # train_dataset = my_dataset(store_path, split_T,split_L, label_T,label_L,data_transform)
    # #test_dataset = my_dataset(store_path, split, name, data_transform)
    # #test_data = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    # batch=32
    # epoch=50
    # use_gpu = torch.cuda.is_available()
    # net=Resnet(blocks=[3,4,6,3],in_size=32,in_ch=3)  #wangluo
    # loss=define_loss()
    # optimizer=define_optimizer()
    # if use_gpu:
    #    net=net.cuda()
    #    loss=loss.cuda()
    # net_path="net.pth"
    # if os.path.exists(net_path):
    #    net=torch.load(net_path)
    #
    # train_data = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=1,drop_last=True)
    # net_path,acc,acc_train,iterator=train(batch,epoch,train_data,net,optimizer,loss)
    # plt.plot(iterator,acc_train,label="train")
    # plt.plot(iterator,acc,label="valid")
    # plt.legend()
    # torch.cuda.empty_cache()
    # test_data = DataLoader(train_dataset,batch_size=32,shuffle=False,num_workers=1,drop_last=True)
    # test(test_data,net_path,loss)
    i=1
    #picture_test=io.imread('C://Users//16334//Desktop//02f7deabfb2f17d4a822d6d4600b4ab909d18fec_raw.jpg')
    #num_picture=np.array(picture_test)
    #num_picture=np.array([picture_test,picture_test,picture_test])
    #num_picture=torch.tensor(num_picture)
    #torch.cat(num_picture,dim=2)
    #out_p=Resnet_18(num_picture,net_path)
