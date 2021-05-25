
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from net_base.util.config import Config
from torch.utils.data import DataLoader
from net_base.util.dataloader import yolo_dataset_collate, YoloDataset
from net_base.nets.yolo_training import YOLOLoss
from net_base.nets.yolo3 import YoloBody



def load_data(annotation_path,Batch_size):
    # 划分数据集，7成训练，3成验证
    val_split = 0.3
    with open(annotation_path) as f:
        lines = f.readlines()
    #打乱数据顺序随机训练
    np.random.seed(1)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)   #验证个数
    num_train = len(lines) - num_val      #训练个数

    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
    train_data = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)
    val_data = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                          drop_last=True, collate_fn=yolo_dataset_collate)
    return train_data,val_data,num_train,num_val

def define_optimizer(lr):
    optimizer=optim.Adam(net.parameters(), lr)
    return optimizer

def define_loss():
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))
    return yolo_losses

def train(net,yolo_losses,epoch,epoch_size,epoch_size_val,train_data,val_data):
    total_loss = 0   #初始化训练误差
    val_loss = 0     #初始化验证误差

    for iteration, batch in enumerate(train_data):
        if iteration >= epoch_size:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        total_loss += loss
        print("epoch:{} iter：{}  total_loss:{}".format(epoch+1,iteration + 1,total_loss.item() / (iteration + 1)))


    for iteration, batch in enumerate(val_data):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
            targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss
        print("epoch:{} iter：{}  val_loss:{}".format(epoch+1,iteration + 1,val_loss.item() / (iteration + 1)))

    torch.save(model.state_dict(), 'logs/pre%d.pth'%((epoch+1)))   #保存pth文件


if __name__ == "__main__":

    Batch_size = 1      #batch大小
    Epoch =50           #训练轮数
    annotation_path = 'training_annotation_final.txt'     #注释
    train_data,val_data,num_train,num_val=load_data(annotation_path,Batch_size)   #划分数据集




    model = YoloBody(Config)                            #初始化网络
    Cuda = True                                         #是否使用GPU加速
    Use_Data_Loader = True                              #是否使用dataloader
    ###模型读取预训练权重文件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #GPU加速
    model_dict = model.state_dict()                                         #模型参数
    pretrained_dict = torch.load("logs/pre.pth", map_location=device)       #读取预训练权重文件
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    net = model.train()    #训练开始

    if Cuda:            #GPU加速
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    lr = 1e-3           #学习率
    optimizer = define_optimizer(lr)    #定义优化器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)  #动态调整学习率
    yolo_losses=define_loss()          #定义loss函数
    iter_size = num_train//Batch_size         #每轮的训练的迭代次数
    iter_size_val = num_val//Batch_size       #每轮的验证迭代次数

    for param in model.backbone.parameters():
        param.requires_grad = False           #训练时冻结已经与训练好的backbone参数

    for epoch in range(Epoch):
        train(net,yolo_losses,epoch,iter_size,iter_size_val,train_data,val_data)
        lr_scheduler.step()
            

