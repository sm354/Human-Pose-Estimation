import sys

from Unet import UNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
torch.set_printoptions(profile="full")

from torch.utils.data import Dataset, DataLoader
from CheckerBoard import *

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import os

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    shape = tensor.shape
    tensor[tensor < 0.5*255] = 0
    tensor[tensor >= 0.5*255] = 255
        # then assign white to the pixel


    print(tensor)
    return Image.fromarray(tensor)

def SavePlots(y1, y2, metric, exp_name):
    try:
        plt.clf()
    except Exception as e:
        pass
    plt.style.use('seaborn')
    plt.title('{}'.format(exp_name))
    plt.xlabel('epochs')
    plt.ylabel(metric)
    epochs=np.arange(1,len(y1)+1,1)
    plt.plot(epochs,y1,label='train{}'.format(metric))
    plt.plot(epochs,y2,label='test{}'.format(metric))
    if metric=='acc':
        ep=np.argmax(y2)
        plt.plot(ep,y2[ep],'r*',label='bestacc@({},{})'.format(ep,y2[ep]))
    plt.legend()
    plt.savefig('{}_{}'.format(exp_name,metric), dpi=95)

def Save_Stats(trainloss, trainacc, testloss, testacc, exp_name):
    data=[]
    data.append(trainloss)
    data.append(testloss)
    data.append(trainacc)
    data.append(testacc)
    data=np.array(data)
    data.reshape((4,-1))
    np.save('{}.npy'.format(exp_name),data)

    SavePlots(trainloss, testloss, 'loss', exp_name)
    SavePlots(trainacc, testacc, 'acc', exp_name)

def train(model, device, train_loader, optimizer, epoch):
    trloss=0
    total_ones=0
    tp = 0
    fp = 0
    fn = 0
    output_ones=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        loss=F.binary_cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        trloss+=loss.item()
        #print(loss.item())
        #print(output)
        #print(target)

        for n in range(target.shape[0]):
            for i in range(target.shape[2]):
                for j in range(target.shape[3]):
                    if target[n,0,i,j]==1:
                        total_ones+=1
                        if output[n,0,i,j]>=0.5:
                            output_ones+=1
        t = Variable(torch.Tensor([0.5])).cuda()  # threshold
        output = (output > t).float() * 1
        tp += (output * target).sum().to(torch.float32)
        fp += ((1 - output) * target).sum().to(torch.float32)
        fn += (output * (1 - target)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    trloss/=(batch_idx+1)
    tracc=100.*f1
    print('TrainLoss={} TrainAcc={}'.format(trloss, tracc))
    return trloss, tracc


def test(model,device,test_loader,best_acc, exp_name, out = False):
    test_loss=0
    total_ones=0
    tp = 0
    fp = 0
    fn = 0
    output_ones=0
    with torch.no_grad():
        k = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            #print(data.shape)
            #print(target.shape)
            output = model(data)
            loss=F.binary_cross_entropy(output.float(),target.float())
            test_loss += loss.item()
            #print(loss.item())
            #print(output)
            #print(target)

            for n in range(target.shape[0]):
                '''if out :
                    data = tensor_to_image(output[n][:][:][0])
                    data.save("{}.png".format(k))
                    k += 1'''
                '''for i in range(target.shape[2]):
                    for j in range(target.shape[3]):
                        if target[n,0,i,j]>=0.7:
                            total_ones+=1
                            if output[n,0,i,j]>=0.7:
                                output_ones+=1'''
            t = Variable(torch.Tensor([0.5])).cuda()  # threshold
            output = (output > t).float() * 1
            tp += (output * target).sum().to(torch.float32)
            fp += ((1 - output) * target).sum().to(torch.float32)
            fn += (output * (1 - target)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    test_loss/=(batch_idx+1)
    #print(output_ones)
    #print(total_ones)
    tstacc=100.*f1

    if tstacc > best_acc:
        torch.save(model.state_dict(), '{}.pth'.format(exp_name))
        best_acc = tstacc

    print('TestLoss={} TestAcc={}'.format(test_loss, tstacc))
    return test_loss, tstacc, best_acc

def main():
    exp_name='1'
    batch_size=10
    test_batch_size=10
    epochs=5
    lr=0.1
    best_acc=float(0)
    
    print("Preparing DATASET --------")
    data_train=GridDataset('train')
    data_test=GridDataset('test')
    kwargs={'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader=torch.utils.data.DataLoader(data_train,batch_size=batch_size,shuffle=True, **kwargs)
    test_loader=torch.utils.data.DataLoader(data_test,batch_size=test_batch_size,shuffle=True, **kwargs)

    print("Prepared DATASET ---------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        print(data)
        print(target)'''
    net = UNet(in_channels=3, n_classes=1, padding=True).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1])
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = 0.9)

    print("Model Prepared; Starting Training ...")
    trloss=[]
    teloss=[]
    trAcc=[]
    teAcc=[]
    for epoch in range(1, epochs + 1):
        print("\n Epoch: {}".format(epoch))
        l,a=train(net, device, train_loader, optimizer, epoch)
        if(epoch == epochs):
            L,A,best_acc=test(net, device, test_loader,best_acc, exp_name, True)
        else:
            L, A, best_acc = test(net, device, test_loader, best_acc, exp_name)
        trloss.append(l)
        trAcc.append(a)
        teloss.append(L)
        teAcc.append(A)
        print(trloss[-1], trAcc[-1])
        print(teloss[-1], teAcc[-1])
	
    Save_Stats(trloss, trAcc.cpu(), teloss, teAcc.cpu(), exp_name)
        
if __name__=='__main__':
    main()
