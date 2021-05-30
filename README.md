先下载权重文件，再运行get_annotation_train.py获得label，最后运行main.py完成定位加分类


pre.pth：  
https://pan.baidu.com/s/1ALLeqNvnoN4L25GDZZJqZQ   
提取码： dzan

net_pred1.pth：  
https://pan.baidu.com/s/1ysIWJkjsje7J5TzlsTN6_w   
提取码： zgbs

net_pred2.pth：  
https://pan.baidu.com/s/1o5MhomQ9FNN5ovGJZEmiyg   
提取码： vc5w

从百度网盘下载三个权重文件后将net_pred1.pth和net_pred2.pth放到与predict.py一个目录下，将pre.pth放到pth文件夹下



get_annotation_train.py/get_annotation_test.py    获得图片的标签和路径

training_annotation.txt/testing_annotation.txt    图片的标签和路径

train.py                                          yolo的训练

predict.py                                        yolo预测图片

classfication.py/ model.py                         二次分类的代码

net_fpn.pth/net_pred1.pth/net_pred2.pth/net_pred3.pth    二次分类的权重文件

pre.pth                                           yolo的权重文件

mian.py                                           综合定位+分类+输出结果

test/data  train/data                             放数据集的文件夹

pth                                               yolo存放权重文件的文件夹

classes                                           存放类别的文件夹

net_base                                          存放相关模型的文件夹

image_show                                        存放yolo检测的图片效果的文件夹

image_composition                                 存放综合定位后的图片效果的文件夹

FPCN                                              存放二次分类代码的文件夹







