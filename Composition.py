# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:12:44 2021

@author: 53412
"""
from predict import show
from PIL import Image,ImageFont, ImageDraw
import get_annotation_test 
import numpy as np
from classification import *
from model import *
import time
import warnings

warnings.filterwarnings("ignore")





def get_center_point(box):
    """
    获得框的中心点和高宽
    """
    center_x=(box[0]+box[1])//2
    center_y=(box[2]+box[3])//2
    h=box[1]-box[0]
    w=box[3]+box[2]
    return center_x,center_y,h,w

def get_box(p1,h,w):
    """
    根据中点和框的高和宽定位框的四个坐标值
    """
    box=[]
    box.append(p1[0]-h//2)
    box.append(p1[0]+h//2)
    box.append(p1[1]-int(w//2))
    box.append(p1[1]+int(w//2))
    return box

def from_image(image):
    """
    从图片转化为向量
    """
    image=image.reshape(1,image.shape[0]*image.shape[1]*image.shape[2])
    vector=np.zeros((1,40000))
    vector[:,:image.shape[1]]=image
    return vector
    
def two_center(p1,p2):
    """
    得到两个点的中点
    """
    x=(p1[0]+p2[0])//2
    y=(p1[1]+p2[1])//2
    return x,y
def prolong_down(p1,p2):
    """
    得到以第二个点为中点向下的延长线上的点
    """
    h=(p2[0]-p1[0])//2
    w=(p2[1]-p1[1])//2
    x=p2[0]+h
    y=p2[1]+w
    return x,y
def prolong_up(p1,p2):
    """
    得到以第二个点为中点向上的延长线上的点
    """
    h=(p2[0]-p1[0])//2
    w=(p2[1]-p1[1])//2
    x=p1[0]-h
    y=p1[1]-w
    return x,y

def check_continue(point_list,h):
    """
    连续性查询，如果每个框的垂直距离小于1.6个框高则认为框是连续的
    """
    continue_=True
    for i in range(len(point_list)-1):
        if point_list[i+1][0]-point_list[i][0]>=1.6*h:
            continue_=False
    return continue_


   
def in_box(point,box):
    """
    判断点是否在框里
    """
    if point[0]>box[0] and point[0]<box[1] and point[1]>box[2] and point[1]<box[3]:
        return True
    else:
        return False
    
def horizon_in_box(point,box):
    """
    判断点在水平维度上是否在框里
    """
    if point[0]>box[0] and point[0]<box[1]:
        return True
    else:
        return False

def match_stack(bottom,T,L):
    """
    从下往上进行匹配堆叠找椎骨的位置
    """
    h=bottom[2]    #高度
    now_bottom=bottom    #当前最低点
    l_num=len(L)          #椎骨数
    t_num=len(T)
    t=-1
    l=-1
    point_list=[]
    is_first=True   #判断是否是第一个点
    for i in range(5):
        get_L=False  #是否找到匹配的椎骨
        while l>=-l_num:
            if (L[l][0]+L[l][1])//2<now_bottom[0] :    #若当前椎骨在当前最低点的上方
                if (L[l][0]+L[l][1])//2>now_bottom[0]-h*1.2 :  #若当前椎骨在当前最低点的上方小于1.2个框高的位置               
                    get_L=True
                    point_list.append([(L[l][0]+L[l][1])//2,(L[l][2]+L[l][3])//2])
                    now_bottom=[(L[l][0]+L[l][1])//2,(L[l][2]+L[l][3])//2]  #更新当前最低点
                    h=L[l][1]-L[l][0]    #更新高度
                    if is_first:        #判断是否是第一次匹配到椎骨和椎间盘
                        is_first=False
                    l-=1
                    break
                else:
                    break
            else:
                l-=1
        if not get_L:   #继续找是否有匹配的椎间盘
            while t>=-t_num:
                if (T[t][0]+T[t][1])//2<now_bottom[0] : #若当前椎间盘在当前最低点的上方
                    if (T[t][0]+T[t][1])//2>now_bottom[0]-h*1 : #若当前椎间盘在当前最低点的上方小于1个框高的位置           
                        get_L=True
                        if is_first:
                            point_list.append([(T[t][0]+T[t][1])//2-h//2,(T[t][2]+T[t][3])//2]) #计算相应椎骨的位置加入点列中
                            now_bottom=[(T[t][0]+T[t][1])//2-h//2,(T[t][2]+T[t][3])//2]  #更新当前最低点
                            is_first=False
                        else:       #若是第一块椎骨则改变椎骨计算方式
                            x=(T[t][0]+T[t][1])//2
                            y=(T[t][2]+T[t][3])//2
                            point_list.append([x-(now_bottom[0]-x)//2,y-(now_bottom[1]-y)//2])
                            now_bottom=[x-(now_bottom[0]-x)//2,y-(now_bottom[1]-y)//2]
                        t-=1
                            
                        break
                    else:
                        break
                else:
                    t-=1
        if not get_L:
            if is_first:    #若是骶骨上的第一个椎骨则要往45度角的方向上堆叠
                now_bottom=[now_bottom[0]-h,now_bottom[1]-h//2]  
                point_list.append(now_bottom)
                is_first=False
            else:     #若没找到匹配的椎骨位置则根据当前位置向上堆叠一个h的高度
                now_bottom=[now_bottom[0]-h,now_bottom[1]]
                point_list.append(now_bottom)
    return point_list


            
def compensate(point_list,num):
    """
    补充缺失的点，向上补充
    """
    length=len(point_list)
    for i in range(num):
        h=point_list[length-2+i][0]-point_list[length-1+i][0]
        w=point_list[length-2+i][1]-point_list[length-1+i][1]
        point_list.append([point_list[length-1+i][0]-h,point_list[length-1+i][1]-w])
    return point_list
    
def in_order_list(point_list):
    """
    对点列里的点进行排序，从上到下的顺序
    """
    length=len(point_list)
    for i in range(length):
            for j in range(i+1,length):
                if point_list[i][0]>point_list[j][0]:
                    point_list[i],point_list[j]=point_list[j],point_list[i]
    return point_list
def in_order_list_down(point_list):
    """
    对点列里的点进行排序，从下到上的顺序
    """
    length=len(point_list)
    for i in range(length):
            for j in range(i+1,length):
                if point_list[i][0]<point_list[j][0]:
                    point_list[i],point_list[j]=point_list[j],point_list[i]
    return point_list
    

def overlap_check(point_list,h):
    """
    重合检测，当组合后的点与定位的点差距小于0.2h时认为两个是同一个点，则删除其中的一个点
    """
    drop=0
    point_list=in_order_list_down(point_list)
    j=0
    for i in range(4):
        if point_list[j][0]-point_list[j+1][0]<0.2*h:
            point_list.pop(j+1)
            drop+=1
        else:
            j=j+1
    point_list=compensate(point_list,drop)
    return in_order_list(point_list)
    
def overlap(p1,p2):
    """
    判断两个点是否属于同一个框
    """
    if pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)<49:
        return True
    else:
        return False
           
        

                
                   
            
    

    
        

class Composition():
    def __init__(self,B,T,L,X):
        self.B=B
        self.L=L
        self.T=T
        self.X=X
        self.get_order_check()
        self.point_list=[]
        self.composite_position()
        self.point_list=in_order_list(self.point_list)
        self.class_=[-1]*11
        self.match_class()
    def get_order_check(self):
        """
        将输入的点列都排好序
        """
        for i in range(len(self.L)):
            for j in range(i+1,len(self.L)):
                if self.L[i][0]>self.L[j][0]:
                    self.L[i],self.L[j]=self.L[j],self.L[i]
        for i in range(len(self.T)):
            for j in range(i+1,len(self.T)):
                if self.T[i][0]>self.T[j][0]:
                    self.T[i],self.T[j]=self.T[j],self.T[i]
        j=0
        
        #若有重合度较大的框则去除后一个
        for i in range(len(self.L)-1):
            if self.L[j+1][0]-self.L[j][0]<0.3*(self.L[j][1]-self.L[j][0]):
                self.L.pop(j+1)
            else:
                j=j+1
        j=0
        for i in range(len(self.T)-1):
            if self.T[j+1][0]-self.T[j][0]<0.3*(self.T[j][1]-self.T[j][0]):
                self.T.pop(j+1)
            else:
                j=j+1
    def match_class(self):
        """
        若最后定位出来的点与原始yolo定位的点位置相近则沿用yolo的分类结果
        """
        for i in range(11):
            if i%2==0:
                for j in range(len(self.T)):
                    if overlap(self.point_list[i],get_center_point(self.T[j])):
                        self.class_[i]=self.T[j][4]
                        self.point_list[i]=get_center_point(self.T[j])[:2]
            else:
                for k in range(len(self.L)):
                    if overlap(self.point_list[i],get_center_point(self.L[k])):
                        self.class_[i]=self.L[k][4]
                        self.point_list[i]=get_center_point(self.L[k])[:2]
                        
    def composite_position(self):
        """
        根据所有的椎骨，椎间盘，骶骨，最大框定出11个点的位置
        """
        if len(self.X)!=0 :   #若骶骨存在
            bottom_message=get_center_point(self.X)    #确定下界
            if len(self.B)!=0 :  #若最大框存在
                if len(self.L)==5:    #若5个椎骨完整
                    if get_center_point(self.L[0])[0]>=self.B[0] and self.L[4][0]<=bottom_message[0] and  check_continue(self.L,bottom_message[2]) and bottom_message[0]-get_center_point(self.L[4])[0]<=1.2*bottom_message[2]: 
                        #若5个椎骨均在最大框中，最低的椎骨位置大于下界不超过1.2个框高，5个椎骨满足连续规则，
                        
                        #操作(1)开头
                        for i in range(5):   #定下5个椎骨的位置
                            self.point_list.append(get_center_point(self.L[i])[:2])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])  #清除重叠的椎骨
                        self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))  #计算最上面的椎间盘
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1])) #定五个椎骨中间4个椎间盘的位置
                        point_last=prolong_down(self.point_list[3],self.point_list[4])   #定最下面的椎间盘
                        if horizon_in_box(point_last,self.T[-1]):  #若最下面的椎间盘预测在原来最后一个椎间盘中
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                        #操作(1)结尾
                            
                    else:
                        #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                        point_list=match_stack(bottom_message,self.T,self.L)
                        for i in range(5):
                            self.point_list.append(point_list[4-i])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])
                        point_first=prolong_up(self.point_list[0],self.point_list[1])  #定最上面的椎间盘
                        if horizon_in_box(point_first,self.T[0]):  #若最上面的椎间盘预测在原来最上面的椎间盘中
                            self.point_list.append(get_center_point(self.T[0])[:2])
                        else:
                            self.point_list.append(point_first)
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                        point_last=prolong_down(self.point_list[3],self.point_list[4])
                        if horizon_in_box(point_last,self.T[-1]):
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                elif len(self.T)==6:  #若6个椎间盘完整
                    if  check_continue(self.T,bottom_message[2])  and bottom_message[0]-get_center_point(self.T[5])[0]<=0.6*bottom_message[2]: 
                        #判断椎间盘的连续性和位置合理性，若合理则直接根据六个椎间盘的中点确定椎骨位置
                        for i in range(6):
                            self.point_list.append(get_center_point(self.T[i])[:2])
                        for i in range(5):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    else:
                        #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                        point_list=match_stack(bottom_message,self.T,self.L)
                        for i in range(5):
                            self.point_list.append(point_list[4-i])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])
                        self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                        point_last=prolong_down(self.point_list[3],self.point_list[4])
                        if horizon_in_box(point_last,self.T[-1]):
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                            
                else:
                    #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                    point_list=match_stack(bottom_message,self.T,self.L)
                    for i in range(5):
                        self.point_list.append(point_list[4-i])
                    self.point_list=overlap_check(self.point_list,bottom_message[2])
                    self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                    for i in range(4):
                        self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    point_last=prolong_down(self.point_list[3],self.point_list[4])
                    # if len(self.T)!=0 and horizon_in_box(point_last,self.T[-1]):
                    #     self.point_list.append(get_center_point(self.T[-1])[:2])
                    # else:
                    #     self.point_list.append(point_last) 
                    self.point_list.append(point_last) 
            else:
                if len(self.L)==5: #若5个椎骨完整
                    if  self.L[4][0]<=bottom_message[0] and  check_continue(self.L,bottom_message[2]) and bottom_message[0]-get_center_point(self.L[4])[0]<=1.2*bottom_message[2]: 
                    #若最低的椎骨位置大于下界不超过1.2个框高，5个椎骨满足连续规则
                        #进行操作（1）
                        for i in range(5):
                            self.point_list.append(get_center_point(self.L[i])[:2])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])
                        self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                        point_last=prolong_down(self.point_list[3],self.point_list[4])
                        if horizon_in_box(point_last,self.T[-1]):
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                    else:
                        #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                        point_list=match_stack(bottom_message,self.T,self.L)
                        for i in range(5):
                            self.point_list.append(point_list[4-i])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])
                        self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                        point_last=prolong_down(self.point_list[3],self.point_list[4])
                        if horizon_in_box(point_last,self.T[-1]):
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                elif len(self.T)==6:
                    if  check_continue(self.T,bottom_message[2])  and bottom_message[0]-get_center_point(self.T[5])[0]<=1.6*bottom_message[2]: 
                        #判断椎间盘的连续性和位置合理性，若合理则直接根据六个椎间盘的中点确定椎骨位置
                        for i in range(6):
                            self.point_list.append(get_center_point(self.T[i])[:2])
                        for i in range(5):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    else:
                        #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                        point_list=match_stack(bottom_message,self.T,self.L)
                        for i in range(5):
                            self.point_list.append(point_list[4-i])
                        self.point_list=overlap_check(self.point_list,bottom_message[2])
                        self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                        for i in range(4):
                            self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                        point_last=prolong_down(self.point_list[3],self.point_list[4])
                        if horizon_in_box(point_last,self.T[-1]):
                            self.point_list.append(get_center_point(self.T[-1])[:2])
                        else:
                            self.point_list.append(point_last)
                            
                else:
                    #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                    point_list=match_stack(bottom_message,self.T,self.L)
                    for i in range(5):
                        self.point_list.append(point_list[4-i])
                    self.point_list=overlap_check(self.point_list,bottom_message[2])
                    self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                    for i in range(4):
                        self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    point_last=prolong_down(self.point_list[3],self.point_list[4])
                    if horizon_in_box(point_last,self.T[-1]):
                        self.point_list.append(get_center_point(self.T[-1])[:2])
                    else:
                        self.point_list.append(point_last)
        else:
            top=get_center_point(self.L[0])
            if len(self.L)==5: #若5个椎骨完整
                if check_continue(self.L,top[2]):
                    #若满足椎骨连续规则
                    for i in range(5):
                        self.point_list.append(get_center_point(self.L[i])[:2])
                    self.point_list=overlap_check(self.point_list,top[2])
                    self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                    for i in range(4):
                        self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    self.point_list.append(prolong_down(self.point_list[3],self.point_list[4]))
            elif len(self.T)==6: #若6个椎间盘完整
                if  check_continue(self.T,top[2])  and get_center_point(self.T[5])[0]-top[0]<=0.6*top[2]: 
                    #判断椎间盘的连续性和位置合理性，若合理则直接根据六个椎间盘的中点确定椎骨位置
                    for i in range(6):
                        self.point_list.append(get_center_point(self.T[i])[:2])
                    for i in range(5):
                        self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                else:
                    #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                    point_list=match_stack([self.B[1],self.B[3],top[2],top[3]],self.T,self.L)
                    for i in range(5):
                        self.point_list.append(point_list[4-i])
                    self.point_list=overlap_check(self.point_list,top[2])
                    self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                    for i in range(4):
                        self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                    point_last=prolong_down(self.point_list[3],self.point_list[4])
                    if horizon_in_box(point_last,self.T[-1]):
                        self.point_list.append(get_center_point(self.T[-1])[:2])
                    else:
                        self.point_list.append(point_last)
            else:
                #先从下到上依次堆叠确定椎骨位置，然后同上操作（1）
                point_list=match_stack([self.B[1],self.B[3],top[2],top[3]],self.T,self.L)
                for i in range(5):
                    self.point_list.append(point_list[4-i])
                self.point_list=overlap_check(self.point_list,top[2])
                self.point_list.append(prolong_up(self.point_list[0],self.point_list[1]))
                for i in range(4):
                    self.point_list.append(two_center(self.point_list[i],self.point_list[i+1]))
                point_last=prolong_down(self.point_list[3],self.point_list[4])
                if horizon_in_box(point_last,self.T[-1]):
                    self.point_list.append(get_center_point(self.T[-1])[:2])
                else:
                    self.point_list.append(point_last)
                
                    
                        
def box_to_composition(box):
    """
    从box分出椎骨，椎间盘，骶骨和最大方框
    """
    b=[]
    l=[]
    t=[]
    x=[]
    b_count=0
    x_count=0
    for index,target in enumerate(box):
        if target[4]<=1:
            l.append(target)
        elif target[4] ==7 and b_count==0:
            b=target
            b_count=1
        elif target[4]==8 and x_count==0:
            x=target
            x_count=1
        else:
            t.append(target)
    return b,l,t,x




time_start=time.time()



box=show()
time_end=time.time()

print("yolo定位加分类的时间："+str(time_end-time_start)+"s")           
            
time_start=time.time()

class_list=[]
total_size=len(box)
for i in range(total_size):
    b,l,t,x=box_to_composition(box[i])
    class_list.append(Composition(b,t,l,x))
    
    
time_end=time.time()
print("综合定位的时间"+str(time_end-time_start)+"s")


test_path="./testing_annotation_final.txt"
with open(test_path) as f:
    lines = f.readlines()
    f.close()
for i in range(total_size):
    if class_list[i].point_list!=[]:
        a=lines[i].split()
        image=a[0]
        image=Image.open(image)
        draw = ImageDraw.Draw(image)
        for point in class_list[i].point_list:
            draw.rectangle(
                    [point[1]- 3, point[0]-3,point[1]+ 3, point[0]+ 3])
        del draw
        image.save("image_composition/image{}.jpg".format(i))
    
real_point=get_annotation_test.get_annotation_test()   #获得真正的定位点位置和类别进行比较
mse=0                                          #均方误差
accu=0                                         #分类正确的点
compos=0                                       #构造出来的点
compos_accu=0                                  #构造出来的点里分类正确的点


time_start=time.time()


all_accu=np.zeros(11)             #每一类的准确率
all_mse=np.zeros(11)              #每一类的均方误差                        


net_pths=('./net_fpn.pth','./net_pred1.pth','./net_pred2.pth','./net_pred3.pth')
net_fpn = torch.load(net_pths[0]).eval()
net_pred1 = torch.load(net_pths[1])
net_pred2 = torch.load(net_pths[2])
net_pred3 = torch.load(net_pths[3])



for i in range(total_size):
    fake_one=[]
    for j in range(11):
        mse+=pow(class_list[i].point_list[j][0]-real_point[i][j][1],2)   #计算水平和竖直距离的平方和
        mse+=pow(class_list[i].point_list[j][1]-real_point[i][j][0],2)
        all_mse[j]+=pow(class_list[i].point_list[j][0]-real_point[i][j][1],2)+pow(class_list[i].point_list[j][1]-real_point[i][j][0],2)
        if j%2==0:   #椎间盘分类
            if class_list[i].class_[j]==real_point[i][j][2]:  #若是原来的点按照yolo的分类结果输出
                all_accu[j]+=1
            if class_list[i].class_[j]==-1:      #若是构造出来的点则放入另一个网络继续分类
                fake_one.append(j)
                compos+=1
        else:    #椎骨分类
            if class_list[i].class_[j]==real_point[i][j][2]: #若是原来的点按照yolo的分类结果输出
                all_accu[j]+=1
            if class_list[i].class_[j]==-1:   #若是构造出来的点则放入另一个网络继续分类
                fake_one.append(j)
                compos+=1

   
    a=lines[i].split()
    image=a[0]
    pred=classification(image,class_list[i].point_list,net_fpn,net_pred1,net_pred2,net_pred3)
    for k in range(len(fake_one)):
        if fake_one[k]%2==0:
            if pred[10-fake_one[k]]+2==real_point[i][fake_one[k]][2]:
                compos_accu+=1
                all_accu[fake_one[k]]+=1
        else:
            if pred[10-fake_one[k]]-5==real_point[i][fake_one[k]][2]:
                compos_accu+=1
                all_accu[fake_one[k]]+=1

   
        
time_end=time.time()
print("二段分类的时间："+str(time_end-time_start)+"s")
                
mse=mse/(11*total_size)   #计算最终的均方误差
rmse=pow(mse,0.5)           #计算均方根误差
print("\n")
for i in range(11):
    accu+=all_accu[i]
    if i%2==0:
        print("椎间盘{}定位均方误差：{:.4f}，均方根误差为：{:.4f}\n分类准确率为：{:.4f}".format(i//2+1,all_mse[i]/total_size,pow(all_mse[i]/total_size,0.5),all_accu[i]/total_size))
    else:
        print("椎骨{}定位均方误差：{:.4f}，均方根误差为：{:.4f}\n分类准确率为：{:.4f}".format(i//2+1,all_mse[i]/total_size,pow(all_mse[i]/total_size,0.5),all_accu[i]/total_size))


print("\n平均均方误差为：{:.4f}，\n平均均方根误差为：{:.4f}\n平均分类准确率为：{:.4f}".format(mse,rmse,(accu+compos_accu)/(11*total_size)))


