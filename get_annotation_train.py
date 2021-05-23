# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:33:26 2021

@author: 53412
"""
# 从所有的dicom文件中找出json文件中若干组序列号相对应的若干个dicom文件，并将它们的标注信息与相应的dicom文件保存在数组中对应起来，其余dicom文件并没有标注信息，无法使用
import os
import json
import glob
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import numpy as np


def get_adjacent_path(path):
    """
    获得对应图片的路径
    """
    s=path[-12:]
    if s[1]=="\\":
        num=eval(path[-5:-4])
        image_pre='\\image{}.dcm'.format(num-1)
        image_next='\\image{}.dcm'.format(num+1)
        return path[:-11]+image_pre,path[:-11]+image_next
    else:
        num=eval(path[-6:-4])
        image_pre='\\image{}.dcm'.format(num-1)
        image_next='\\image{}.dcm'.format(num+1)
        return path[:-12]+image_pre,path[:-12]+image_next

def get_target(points):
    """
    获得对应标记点的位置，类别和疾病类别
    """
    point=[]
    class_=[]
    type_=[]
    for i in range(11):
        p=points[i].split(",{")
        point_=p[0].split(",")
        point.append([eval(point_[0]),eval(point_[1])])
        p=eval("{"+p[1])
        type_.append(p['identification'])
        if 'disc' in p.keys() and p['disc']!='':
            class_.append(p['disc'])
        else:
            class_.append(p['vertebra'])
    return point,class_,type_

def point_to_box(p1,p2,p3):
    """
    利用三个点来画出椎骨目标的框
    """
    h=p3[1]-p1[1]    #框高
    w=h*1.4          #框宽
    x1=int(p2[1]-h/2)
    x2=int(p2[1]+h/2)
    y1=int(p2[0]-w/2)
    y2=int(p2[0]+w/2)
    return x1,x2,y1,y2   #返回框的4个坐标

def point_to_box2(p1,p2,p3):
    """
     利用三个点来画出椎间盘目标的框
    """
    h=p3[1]-p1[1]
    w=h*1.4
    h=h/1.25       #考虑椎间盘的高度小于椎骨
    x1=int(p2[1]-h/2)
    x2=int(p2[1]+h/2)
    y1=int(p2[0]-w/2)
    y2=int(p2[0]+w/2)
    return x1,x2,y1,y2

def get_first_final_box(p1,h,w):
    """
    根据一个点的坐标和高宽返回目标框的四个坐标
    """
    x1=int(p1[1]-h/2)
    x2=int(p1[1]+h/2)
    y1=int(p1[0]-w/2)
    y2=int(p1[0]+w/2)
    return x1,x2,y1,y2
      
def get_box(point,class_,type_):
    """
    获得目标检测的label：框的四个坐标和类别
    """
    disc_index=[]       #椎间盘索引
    vertebra_index=[]   #椎骨索引
    annotation_text=[]  #打标签的文字txt内容
    boxes=[]            #保存框坐标
    disc_index.append(type_.index("T12-L1"))      #找到相应椎间盘和椎骨的索引
    disc_index.append(type_.index("L1-L2"))
    disc_index.append(type_.index("L2-L3"))
    disc_index.append(type_.index("L3-L4"))
    disc_index.append(type_.index("L4-L5"))
    disc_index.append(type_.index("L5-S1"))
    vertebra_index.append(type_.index("L1"))
    vertebra_index.append(type_.index("L2"))
    vertebra_index.append(type_.index("L3"))
    vertebra_index.append(type_.index("L4"))
    vertebra_index.append(type_.index("L5"))

    for i in range(5):
        x1,x2,y1,y2=point_to_box(point[disc_index[i]],point[vertebra_index[i]],point[disc_index[i+1]])   #获得椎骨框的四个点
        cla=class_[vertebra_index[i]][-1]                                                                #获得椎骨类别
        annotation_text.append(" {},{},{},{},{}".format(y1,x1,y2,x2,eval(cla)-1))                        #txt的label注释文字
        boxes.append([x1,x2,y1,y2])
    x1,x2,y1,y2=get_first_final_box(point[disc_index[0]],boxes[0][1]-boxes[0][0],boxes[0][3]-boxes[0][2])   #计算第一个椎间盘框的四个坐标
    cla=class_[disc_index[0]][-1]
    annotation_text.append(" {},{},{},{},{}".format(y1,x1,y2,x2,eval(cla)+1))
    boxes.append([x1,x2,y1,y2])
    
    
    x1,x2,y1,y2=get_first_final_box(point[disc_index[5]],boxes[4][1]-boxes[4][0],boxes[4][3]-boxes[4][2])   #计算最后一个椎间盘框的四个坐标
    cla=class_[disc_index[5]][-1]
    annotation_text.append(" {},{},{},{},{}".format(y1,x1,y2,x2,eval(cla)+1))
    boxes.append([x1,x2,y1,y2])
    
    for i in range(4):
        x1,x2,y1,y2=point_to_box2(point[vertebra_index[i]],point[disc_index[i+1]],point[vertebra_index[i+1]])   #获得椎间盘框的四个坐标
        cla=class_[disc_index[i]][-1]
        annotation_text.append(" {},{},{},{},{}".format(y1,x1,y2,x2,eval(cla)+1))
        boxes.append([x1,x2,y1,y2])
    
    big_box = np.array(boxes)  #将坐标列表转化为numpy方便计算最值
    biggest = []              #将所有椎骨和椎间盘框住的大方格
    biggest.append(big_box.min(axis=0)[0])    #获得四个方位的坐标
    biggest.append(big_box.max(axis=0)[1])
    biggest.append(big_box.min(axis=0)[2])
    biggest.append(big_box.max(axis=0)[3])


    h = boxes[6][1] - boxes[6][0]
    w = boxes[6][3] - boxes[6][2]
    sacrum = []                    #记录骶骨的数据方便定位
    sacrum.append(boxes[6][0] + h // 2)   #骶骨的位置大约在最后一个椎间盘右下方45度处
    sacrum.append(boxes[6][1] + h // 2)
    sacrum.append(boxes[6][2] )
    sacrum.append(boxes[6][3] + w // 2)
    return annotation_text, boxes, biggest,sacrum

def cat_(img1,img2,img3):
    """
    将相同大小的三张灰度图片叠成三通道的图片
    :param img1: 中间的图片
    :param img2: 前一张图片
    :param img3: 后一张图片
    :return: 三张灰度图堆积成为彩色图片
    """
    if img1.shape==img2.shape:     #如果前一张与中间的图片大小相同
        if img1.shape==img3.shape:   #如果后一张与中间的图片大小相同
            return np.concatenate((img1,img2,img3),axis=2)
        else:
            return np.concatenate((img1,img1,img2),axis=2)
    elif img1.shape==img3.shape:
        return np.concatenate((img1,img1,img3),axis=2)
    else:
        return np.concatenate((img1,img1,img1),axis=2)
        
def get_accurate_point(point,class_,type_):
    disc_index=[]
    vertebra_index=[]
    annotation=[]
    disc_index.append(type_.index("T12-L1"))
    disc_index.append(type_.index("L1-L2"))
    disc_index.append(type_.index("L2-L3"))
    disc_index.append(type_.index("L3-L4"))
    disc_index.append(type_.index("L4-L5"))
    disc_index.append(type_.index("L5-S1"))
    vertebra_index.append(type_.index("L1"))
    vertebra_index.append(type_.index("L2"))
    vertebra_index.append(type_.index("L3"))
    vertebra_index.append(type_.index("L4"))
    vertebra_index.append(type_.index("L5"))
    for i in range(11):
        if i%2==0:
            annotation.append([point[disc_index[i//2]][0],point[disc_index[i//2]][1],
                              eval(class_[disc_index[i//2]][-1])+1])
        else:
            annotation.append([point[vertebra_index[i//2]][0],point[vertebra_index[i//2]][1],
                              eval(class_[vertebra_index[i//2]][-1])-1])
    return annotation       


    return im
if __name__ == '__main__':


    
    f_=open("training_annotation_final.txt","w")
    annotation=[]
    train_data=r".\train\data"
    test_data_path=r".\test\data"
    for filename in os.listdir(train_data):
        if os.path.splitext(filename)[1] == '.txt':
            f=open(train_data+'\\'+filename)
            if filename=="study82.txt" or filename=="study164.txt" or filename=="study77.txt":
                continue
            annotation=f.readlines()
            point_,class_,type_=get_target(annotation)
            text,boxes,biggest,sacrum=get_box(point_,class_,type_)  #获得目标检测框的数据
            str1=(train_data+'\\'+filename).replace('\\', '/')  #保存图片的路径方便训练时读取
            str1=str1.replace(".txt",".jpg")
            f_.write(str1+" {},{},{},{},{}".format(biggest[2],biggest[0],biggest[3],biggest[1],7))  #写入最大的框的标注
            f_.write(" {},{},{},{},{}".format(sacrum[2], sacrum[0], sacrum[3], sacrum[1], 8))     #写入骶骨的框的标注
            for j in range(11):
                f_.write(text[j])
            if filename!="study99.txt":
                f_.write("\n")
            f.close()
    f_.close()
    #
    f_=open("testing_annotation_final.txt","w")
    annotation=[]
    train_data=r".\train\data"
    test_data=r".\test\data"
    for filename in os.listdir(test_data):
        if os.path.splitext(filename)[1] == '.txt':
            f=open(test_data+'\\'+filename)
            annotation=f.readlines()
            point_,class_,type_=get_target(annotation)
            text,boxes,biggest,sacrum=get_box(point_,class_,type_)  #获得目标检测框的数据
            str1=(test_data+'\\'+filename).replace('\\', '/')  #保存图片的路径方便训练时读取
            str1=str1.replace(".txt",".jpg")
            f_.write(str1+" {},{},{},{},{}".format(biggest[2],biggest[0],biggest[3],biggest[1],7))  #写入最大的框的标注
            f_.write(" {},{},{},{},{}".format(sacrum[2], sacrum[0], sacrum[3], sacrum[1], 8))     #写入骶骨的框的标注
            for j in range(11):
                f_.write(text[j])
            if filename!="study9.txt":
                f_.write("\n")
            f.close()
    f_.close()
        



