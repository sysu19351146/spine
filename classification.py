from PIL import Image
import numpy as np
from model import *
from torch.autograd import Variable
import torch.nn.functional as F
citar_1,citar_2,citar_3=1,1,1

def functional_conv2d(im):
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
def classification(pth,loc,net_fpn,net_pred1,net_pred2,net_pred3):
    img=np.array(Image.open(pth))
    if len(img.shape)==3:
        img=img[:,:,0]
        img=np.squeeze(img)
    x=[]
    for i in range(10,-1,-1):
        y1 = max(loc[i][1] - 80, 0)
        y2 = min(loc[i][1] + 80, 512)
        x1 = max(loc[i][0] - 40, 0)
        x2 = min(loc[i][0] + 40, 512)
        z=img[y1:y2,x1:x2]
        if np.size(z,0)!=160 or np.size(z,1)!=80:
            z=np.resize(z,(160,80))
        x.append(z)
    x = np.array(x)
    x = torch.from_numpy(x).double()
    x=x.unsqueeze(dim=0)
    # net_fpn = torch.load(net_pths[0]).eval()
    # net_pred1 = torch.load(net_pths[1])
    # net_pred2 = torch.load(net_pths[2])
    # net_pred3 = torch.load(net_pths[3])
    x = functional_conv2d(x)
    # total = sum([param.nelement() for param in net_fpn.parameters()])
    # total=total+sum([param.nelement() for param in net_pred1.parameters()])   #计算模型参数量
    # total=total+sum([param.nelement() for param in net_pred2.parameters()])
    # total=total+sum([param.nelement() for param in net_pred3.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    y1,y2,y3=net_fpn(x)
    y_pred = citar_1 * net_pred1(y1) + citar_2 * net_pred2(y2) + citar_3 * net_pred3(y3)
    y_pred=y_pred.view(11,7)
    y_pred = torch.argmax(y_pred, dim=-1)
    return y_pred.numpy()

# if __name__ == '__main__':
#     loc=[[138,62],[137,72],[135,83],[133,95],[130,106],[129,118],[127,130],[126,145],[127,157],[127,171],[130,182]]#从T12-L1到L5-S1
#     z=classification('./image0.jpg',loc,0)
#     print(z)
