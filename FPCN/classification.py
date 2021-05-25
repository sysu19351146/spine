from PIL import Image
import numpy as np
from model import *
from torch.autograd import Variable
import torch.nn.functional as F
citar_1,citar_2,citar_3=1,1,1

def functional_conv2d(im):#sobel算子提取边缘特征
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

#输入图片的位置，椎间盘和椎骨的坐标，和要查询的椎骨或椎间盘序号，输出判断 0-4代表椎间盘的v1-v5，5-6代表椎骨的v1-v2
def classification(pth,loc,num,net_pths=('./net_fpn.pth','./net_pred1.pth','./net_pred2.pth','./net_pred3.pth')):
    #读入数据
    img=np.array(Image.open(pth))
    if len(img.shape)==3:
        img=img[:,:,0]
        img=np.squeeze(img)
    x=[]
    #截取竖直方向160个像素，水平方向80个像素作为输入
    for i in range(10,-1,-1):
        y1 = max(loc[i][1] - 80, 0)
        y2 = min(loc[i][1] + 80, 512)
        x1 = max(loc[i][0] - 40, 0)
        x2 = min(loc[i][0] + 40, 512)
        z=img[y1:y2,x1:x2]
        # 如果取样越界了就resize为160*80
        if np.size(z,0)!=160 or np.size(z,1)!=80:
            z=np.resize(z,(160,80))
        x.append(z)
    x = np.array(x)
    x = torch.from_numpy(x).double()
    x=x.unsqueeze(dim=0)
    #读取网络
    net_fpn = torch.load(net_pths[0]).eval()
    net_pred1 = torch.load(net_pths[1])
    net_pred2 = torch.load(net_pths[2])
    net_pred3 = torch.load(net_pths[3])
    #进行sobel算子边缘检测
    x = functional_conv2d(x)
    #获取3个层次的特征
    y1,y2,y3=net_fpn(x)
    #将预测结果加权
    y_pred = citar_1 * net_pred1(y1) + citar_2 * net_pred2(y2) + citar_3 * net_pred3(y3)
    y_pred=y_pred.view(11,7)
    #取预测的概率值最大的类作为结果
    y_pred = torch.argmax(y_pred, dim=-1)
    result=y_pred[10-num]
    return result.numpy()
