from net_base.yolo import YOLO
from PIL import Image



#使用yolo模型对图片进行检测
def show():
    yolo = YOLO(model_path="pth/pre.pth",classes_path="classes/new_classes.txt",max_box=13)
    test_path="./testing_annotation_final.txt"
    with open(test_path) as f:
        lines = f.readlines()
        f.close()
    i=-1
    boxes=[]
    for line in lines:
        i+=1
        a=line.split()
        image=a[0]
        image=Image.open(image)
        r_image,text,box = yolo.detect_image(image)
        boxes.append(box)
        r_image.save("./image_show/image{}.jpg".format(i))
    return boxes
# show();

