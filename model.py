import torch
import math


seed = 0


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class inception(torch.nn.Module):
    def __init__(self,in_chan):
        super(inception,self).__init__()
        div=4
        self.con_1_1=torch.nn.Conv2d(in_chan,int(in_chan/div),1)
        self.con_1_2 = torch.nn.Conv2d(int(in_chan / div), int(in_chan / div), 7, padding=(3,3))
        self.con_2_1 = torch.nn.Conv2d(in_chan,int(in_chan/div), 1)
        self.con_2_2 = torch.nn.Conv2d(int(in_chan/div),int(in_chan/div),3,padding=(1,1))
        self.con_3_1 = torch.nn.Conv2d(in_chan, int(in_chan/div), 1)
        self.con_3_2 = torch.nn.Conv2d(int(in_chan/div),int(in_chan/div),5,padding=(2,2))
        self.pool = torch.nn.MaxPool2d(kernel_size=(3, 3), padding=(1,1),stride=1)
        self.con_4 = torch.nn.Conv2d(in_chan,int(in_chan/div),1)
    def forward(self,x):
        y1=self.con_1_2(self.con_1_1(x))
        y2=self.con_2_2(self.con_2_1(x))
        y3=self.con_3_2(self.con_3_1(x))
        y4=self.con_4(self.pool(x))
        y=torch.cat([y1,y2,y3,y4],dim=1)
        return y

class FPN(torch.nn.Module):
    def __init__(self,chan):
        super(FPN, self).__init__()
        self.pre_con=torch.nn.Conv2d(11,chan,kernel_size=(3,3),padding=(1,1))
        #提取特征
        self.conv1=inception(chan)
        self.act_1 =torch.nn.LeakyReLU(0.1)#sigmoid
        self.pool1 =torch.nn.MaxPool2d((2,2),stride=(2,2))
        self.dr_1=torch.nn.Dropout(0.5)

        self.conv2 = inception(chan)
        self.act_2 = torch.nn.LeakyReLU(0.1)
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dr_2 = torch.nn.Dropout(0.5)

        self.conv3 = inception(chan)
        self.act_3 = torch.nn.LeakyReLU(0.1)
        self.dr_3 = torch.nn.Dropout(0.5)

        self.con1_1_first=torch.nn.Conv2d(chan,int(chan/2),(1,1),padding=0)#横向的1*1
        self.con1_1_second = torch.nn.Conv2d(chan,int(chan/2), (1, 1), padding=0)

        self.upsample_1=torch.nn.Upsample(scale_factor=2,mode='bilinear')#纵向的上采样，用双线性好一点,mode='bilinear'
        self.upsample_2 =torch.nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):
        x=self.pre_con(x)
        y1=self.act_1(self.conv1(x))
        y1=self.dr_1(y1)
        y1_after_pool=self.pool1(y1)
        y2=self.act_2(self.conv2(y1_after_pool))
        y2=self.dr_2(y2)
        y2_after_pool=self.pool2(y2)
        y3=self.act_3(self.conv3(y2_after_pool))
        y3=self.dr_3(y3)

        y3_out=y3#通道为32
        y2_out = torch.cat((self.upsample_2(y3_out), self.con1_1_second(y2)), 1)#通道为40
        y1_out = torch.cat((self.upsample_1(y2_out), self.con1_1_first(y1)), 1)#通道为48
        return y1_out,y2_out,y3_out

class Pred(torch.nn.Module):
    def __init__(self,in_size):
        super(Pred, self).__init__()
        C, H, W = in_size[:]
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        t = int(abs(math.log(C, 2) + 1) / 2)
        k = t if t % 2 else t + 1
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        self.in_chan=C
        self.in_size=in_size
        self.conv_1=torch.nn.Conv2d(in_channels=C,out_channels=2*C,kernel_size=3,padding=1)
        self.relu=torch.nn.ReLU()
        self.pooling=torch.nn.MaxPool2d((2,2),stride=(2,2))
        self.conv_2=torch.nn.Conv2d(in_channels=2*C,out_channels=4*C,kernel_size=3,padding=1)
        self.fc=torch.nn.Linear(int(C*H*W),11*7)
    def forward(self,x):
        w=torch.squeeze(self.gap(x),dim=-1).permute(0, 2, 1)
        w=self.conv(w)
        w=w.transpose(-1,-2).unsqueeze(-1)
        x=x*w.expand_as(x)
        y1=self.conv_1(x)
        y2=self.pooling(self.relu(y1))
        y3=self.conv_2(y2)
        y3=y3.view(x.size()[0],int(self.in_chan*self.in_size[1]*self.in_size[2]))
        y4=self.fc(y3)
        return y4