#用于在训练集和测试集上测试效果
from fpcn_train import *
import torch
def test(x,y,net_pths,num):
    acc=0
    p='./'
    net_fpn = torch.load(p+net_pths[0]).eval()
    net_pred1=torch.load(p+net_pths[1])
    net_pred2=torch.load(p+net_pths[2])
    net_pred3=torch.load(p+net_pths[3])
    li=torch.linspace(num*10,num*10+9,steps=10).long()#每次对10幅图进行测试
    x_in=x[li,...]
    y_select=y[li,...]
    y1,y2,y3=net_fpn(x_in)#提取特征
    pred=citar_1*net_pred1(y1)+citar_2*net_pred2(y2)+citar_3*net_pred3(y3)#加权预测
    pred=pred.view(pred.size()[0],11,7)
    pred=torch.argmax(pred,dim=2)
    for i in range(pred.size()[0]):
        for j in range(pred.size()[1]):
            if pred[i,j]==y_select[i,j]:
                acc=acc+1
    acc=acc/pred.numel()
    print('pred:', pred.numpy())
    print('GT:',y_select.numpy())
    return acc  #返回10幅图的准确率



if __name__=='__main__':
    head_pth = './脊柱疾病智能诊断/'
    split=['train','test']
    n=1
    pth=head_pth+split[n]+'/data'
    x, y,num_of_data = load_data(pth)
    x = x.astype(float)
    y = y.astype(int)
    x = torch.from_numpy(x).double()
    y = torch.from_numpy(y).int()
    x = functional_conv2d(x)
    pth_of_model = './'
    files = os.listdir(pth_of_model)

    if all(word if word in files else False for word in net_pths):
        print('start testing')
        all_acc=0
        times=int(x.size()[0]/10)
        for i in range(times):
            z=test(x, y, net_pths,i)
            all_acc+=z
        print(all_acc/times)
    else:
        print('Please train the model first')