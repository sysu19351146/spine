# 从所有的dicom文件中找出json文件中若干组序列号相对应的若干个dicom文件，并将它们的标注信息与相应的dicom文件保存在数组中对应起来，其余dicom文件并没有标注信息，无法使用
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import numpy as np


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



def get_annotation_test():
    accurate_point=[]
    ftest=open("testing_annotation_final.txt")
    k=ftest.readlines()
    for filename in k:
        file=filename.split()
        file=file[0]
        file=file.replace('.jpg', '.txt')
        f=open(file)
        annotation=f.readlines()
        point_,class_,type_=get_target(annotation)
        accurate=get_accurate_point(point_,class_,type_)
        accurate_point.append(accurate)
    return accurate_point
    

    
    
    
