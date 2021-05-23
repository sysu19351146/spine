#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image



# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()


def show():
    yolo = YOLO(model_path="logs/pre.pth",classes_path="model_data/new_classes.txt",max_box=13)
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
        # r_image.save("./image_show/image{}.jpg".format(i))
    return boxes
# show();

