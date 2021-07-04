import os
import sys
import argparse
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils.CheckerBoard import *
from models.Unet import UNet
from models.resUnet_plpl import ResUnetPlusPlus
from models.attenUnet import AttU_Net

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    shape = tensor.shape
    tensor[tensor < 0.5*255] = 0
    tensor[tensor >= 0.5*255] = 255

    # print(tensor)
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
    plt.plot(epochs,y2,label='val{}'.format(metric))
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
        #print(type(output))
        #print(type(target))
        #print(output.shape)
        #print(target.shape)
        loss=F.binary_cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        trloss+=loss.item()
        #if batch_idx % 100 == 0:
        #print(loss.item())
        
        '''if(batch_idx % 100 == 0):
            print("------------------------------------DATA-----------------------------------")
            #print(data.shape)
            print(data)
            print("------------------------------------OUTPUT-----------------------------------")
            #print(output.shape)
            print(output)
            print("------------------------------------TARGET-----------------------------------")
            #print(target.shape)
            print(target)'''

        t = Variable(torch.Tensor([0.5])).cuda()  # threshold
        k = Variable(torch.Tensor([1])).cuda()  # threshold
        total_ones = (target > t).sum()
        output_ones = (output > t).sum()
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
    #print(f'Output Ones: {output_ones}, Total Ones: {total_ones}')
    return trloss, tracc


def test(model,device,val_loader,best_acc, scheduler, exp_name, out = True):
    test_loss=0
    total_ones=0
    tp = 0
    fp = 0
    fn = 0
    output_ones=0
    output_im = []
    with torch.no_grad():
        k = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            #print(data.shape)
            #print(target.shape)
            output = model(data)
            loss=F.binary_cross_entropy(output.float(),target.float())
            test_loss += loss.item()
            #print(loss.item())
            #print(output)
            #print(target)
            k = output * 255
            k = np.array(k.cpu(), dtype = np.uint8)
            #print(k.shape)
            #print(k[0].shape)
            for i in range(len(k)):
                output_im.append(k[i])

            t = Variable(torch.Tensor([0.5])).cuda()  # threshold
            k = Variable(torch.Tensor([1])).cuda()  # threshold
            total_ones = (target > t).sum()
            output_ones = (output > t).sum()
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

    if(out):
        scheduler.step(test_loss)
    
    if tstacc > best_acc:
        torch.save(model.module.state_dict(), '{}.pth'.format(exp_name))
        best_acc = tstacc

    print('TestLoss={} TestAcc={}'.format(test_loss, tstacc))
    #print(f'Output Ones: {output_ones}, Total Ones: {total_ones}')
    return test_loss, tstacc, best_acc

def main():
    parser = argparse.ArgumentParser()
    print("hi")
    parser.add_argument("-n", "--one", required = True, help = "Name of experiment")
    parser.add_argument("-a", "--two", required = True, help = "Address of data")
    parser.add_argument("-e", "--three", required = True, help = "Number of epochs")
    parser.add_argument("-m", "--four", required = True, help = "Choose the model:\n 1 for Unet, 2 for ResUnet++, 3 for Attention Unet.")
    print("parsing")
    args = vars(parser.parse_args())
    print("parsed")
    print(args)
    exp_name=args['one']
    batch_size=50
    test_batch_size=50
    epochs=int(args['three'])
    lr=0.01
    best_acc=float(0)
    
    print("Preparing DATASET --------")
    data_train = CheckerBoard('Train', args['two'])
    data_test = CheckerBoard('Val', args['two'])
    kwargs={'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader=torch.utils.data.DataLoader(data_train,batch_size=batch_size,shuffle=False, **kwargs)
    val_loader=torch.utils.data.DataLoader(data_test,batch_size=test_batch_size,shuffle=False, **kwargs)

    print("Prepared DATASET ---------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        print(data)
        print(target)'''
    if(args['four'] == 1):
        net = UNet(in_channels = 3, n_classes = 1).to(device)
    if(args['four'] == 2):
        net = ResUnetPlusPlus(channel = 3).to(device)
    if(args['four'] == 3):
        net = AttU_Net(img_ch = 3, out_ch=1).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1])
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    print("Model Prepared; Starting Training ...")
    trloss=[]
    teloss=[]
    trAcc=[]
    teAcc=[]
    for epoch in range(1, epochs + 1):
        print("\n Epoch: {}".format(epoch))
        l,a=train(net, device, train_loader, optimizer, epoch)
        if(epoch == epochs):
            L,A,best_acc=test(net, device, val_loader,best_acc, scheduler, exp_name)
        else:
            L, A, best_acc = test(net, device, val_loader, best_acc, scheduler, exp_name)

        trloss.append(l)
        trAcc.append(a.cpu())
        teloss.append(L)
        teAcc.append(A.cpu())
        #print(trloss[-1], trAcc[-1])
        #print(teloss[-1], teAcc[-1])
	
    Save_Stats(trloss, trAcc, teloss, teAcc, exp_name)

    testD = CheckerBoard("Test", args['two'])
    test_loader = torch.utils.data.DataLoader(testD, batch_size=batch_size, shuffle=False, **kwargs)
    net.module.load_state_dict(torch.load(f'./{exp_name}.pth'))
    L, A, acc = test(net, device, test_loader, best_acc, scheduler, exp_name, False)

    print(f'Test Set Accuracy: {A.cpu()}, Loss: {L}')
        
if __name__=='__main__':
    main()
    print("Completed!")
