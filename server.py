
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import models
import os
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
import random
import torchvision.models as models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', nargs='+', type=str, default='0,1,2,3,4,5', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-d', '--distribution', nargs='+', type=str, default='1,1,8', help='models type to use(e.g. 1 vgg13, 1 vgg16, 8 vgg19)')
parser.add_argument('-f', '--fusion', type=int, default=1, help='fusion or not')
parser.add_argument('-op', '--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('-lr', '--learningrate', nargs='+',type=float, default=0.001, help='learning rate')
parser.add_argument('-td', '--trainingdata', type=str, help='training data')
parser.add_argument('-se', '--startepoch', type=int, help='start epoch')
parser.add_argument('-ui', '--uploadinterval', type=int, help='upload interval')
parser.add_argument('-p', '--pretrained', type=int, help='pretrained')
parser.add_argument('-e', '--epoch', type=int, help='epoch')
parser.add_argument('-t', '--type', type=str, help='type')
parser.add_argument('-ha', '--half_num', type=int, help='half_num')
parser.add_argument('-ir', '--innerratio', type=float, default=1.0, help='innerratio')
parser.add_argument('-or', '--outerratio', type=float, default=1.0, help='outerratio')
parser.add_argument('-m', '--mode', type=str, default='origin', help='mode:origin,flop,hermes')
parser.add_argument('-i', '--iid', type=int, default='iid', help='iid or non-iid')
                        

def expansion(global_params,client_params,basiclayer,net_index):
    for key,index in zip(basiclayer,range(len(global_params))):
        shape=list(client_params[key].shape)
        
        if(len(shape)==4):
            global_params[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]+=client_params[key].cpu()*local_conv_index[net_index][key]
        elif(len(shape)==2):
            global_params[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1]]+=client_params[key].cpu()*local_conv_index[net_index][key]
        elif(len(shape)==1):
            global_params[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0]]+=client_params[key].cpu()*local_conv_index[net_index][key]
        else:
            global_params[index]+=client_params[key].cpu()*local_conv_index[net_index][key]
    return global_params

def compression(client_params,net_index):
    global_params={}
    index=0
    for key in client_params.keys():
        if(key in basic_layer[net_index]):
            shape=list(client_params[key].shape)
            if(len(shape)==4):
                global_params[key]=global_model[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]
            elif(len(shape)==2):
                global_params[key]=global_model[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1]]
            elif(len(shape)==1):
                global_params[key]=global_model[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0]]
            else:
                global_params[key]=global_model[index]
            index+=1
        else:
            global_params[key]=client_params[key]
    return global_params

def layoffset_generator():
    layer_offset=[]
    for net_index in range(len(Net_type)):
        temp_offset=[]
        for offset_range in layer_offset_range[net_index]:
            if(len(offset_range)==4):
                temp_offset.append([random.randint(offset_range[0],offset_range[1]),random.randint(offset_range[2],offset_range[3])])
            elif(len(offset_range)==2):
                temp_offset.append([random.randint(offset_range[0],offset_range[1])])
            else:
                temp_offset.append([])
        layer_offset.append(temp_offset)
    return layer_offset
    
def validation(testset,net,device,net_index):
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=1)
    correct = 0
    total = 0
    test_net=net.eval()
    test_net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = test_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    overall=( 100 * correct / total)
    if args.iid==0:
        correct = 0
        total = 0
        testloader = torch.utils.data.DataLoader(data_subset_test[net_index], batch_size=4,shuffle=False, num_workers=1)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = test_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    test_net.to("cpu")
    return overall,( 100 * correct / total)

def global_shape_generator():
    global_shape=[]
    for i in range(len(Net)):
        net_dict=Net[i].state_dict()
        for j in range(len(basic_layer[i])):
            if(len(net_dict[basic_layer[i][j]].shape)==4):
                if i==0:
                    global_shape.append(list(net_dict[basic_layer[i][j]].shape))
                else:
                    if global_shape[j][0]<net_dict[basic_layer[i][j]].shape[0]:
                        global_shape[j][0]=net_dict[basic_layer[i][j]].shape[0]
                    if global_shape[j][1]<net_dict[basic_layer[i][j]].shape[1]:
                        global_shape[j][1]=net_dict[basic_layer[i][j]].shape[1]
                    if global_shape[j][2]<net_dict[basic_layer[i][j]].shape[2]:
                        global_shape[j][2]=net_dict[basic_layer[i][j]].shape[2]
                    if global_shape[j][3]<net_dict[basic_layer[i][j]].shape[3]:
                        global_shape[j][3]=net_dict[basic_layer[i][j]].shape[3]
            elif(len(net_dict[basic_layer[i][j]].shape)==2):
                if i==0:
                    global_shape.append(list(net_dict[basic_layer[i][j]].shape))
                else:
                    if global_shape[j][0]<net_dict[basic_layer[i][j]].shape[0]:
                        global_shape[j][0]=net_dict[basic_layer[i][j]].shape[0]
                    if global_shape[j][1]<net_dict[basic_layer[i][j]].shape[1]:
                        global_shape[j][1]=net_dict[basic_layer[i][j]].shape[1]
            elif(len(net_dict[basic_layer[i][j]].shape)==1):
                if i==0:
                    global_shape.append(list(net_dict[basic_layer[i][j]].shape))
                else:
                    if global_shape[j][0]<net_dict[basic_layer[i][j]].shape[0]:
                        global_shape[j][0]=net_dict[basic_layer[i][j]].shape[0]
            else:
                if i==0:
                    global_shape.append(list(net_dict[basic_layer[i][j]].shape))
    
    return global_shape

def layer_offset_range_generator():
    layer_offset_range=[]
    for i in range(len(Net)):
        node_offset=[]
        net_dict=Net[i].state_dict()
        for j in range(len(basic_layer[i])):
            if(len(net_dict[basic_layer[i][j]].shape)==4):
                node_offset.append([0,global_shape[j][0]-net_dict[basic_layer[i][j]].shape[0],0,global_shape[j][1]-net_dict[basic_layer[i][j]].shape[1]])
            elif(len(net_dict[basic_layer[i][j]].shape)==2):
                node_offset.append([0,global_shape[j][0]-net_dict[basic_layer[i][j]].shape[0],0,global_shape[j][1]-net_dict[basic_layer[i][j]].shape[1]])
            elif(len(net_dict[basic_layer[i][j]].shape)==1):
                node_offset.append([0,global_shape[j][0]-net_dict[basic_layer[i][j]].shape[0]])   
            else:
                node_offset.append([])
        layer_offset_range.append(node_offset)
    return layer_offset_range

args = parser.parse_args()
gpu1="cuda:"+args.gpu[0]
gpu2="cuda:"+args.gpu[1]
'''gpu3="cuda:"+args.gpu[2]
gpu4="cuda:"+args.gpu[3]'''
print('gpu',args.gpu)#,gpu2,gpu3,gpu4)
print('distribution',args.distribution)
print('fusion',args.fusion)
print('optimizer',args.optimizer)
print('learningrate',args.learningrate)
print('training data',args.trainingdata)
print('start epoch', args.startepoch)
print('upload interval', args.uploadinterval)
print('pretrained', args.pretrained)
print('epoch',args.epoch)
print('type',args.type)
print('half_num',args.half_num)
print('innerratio',args.innerratio)
print('outerratio',args.outerratio)
print('mode',args.mode)
print('iid or non-iid',args.iid)
'''
#resnet

basic_layer=[['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']]

global_shape=[[64, 3, 7, 7], [64, 64, 3, 3], [64, 64, 3, 3], [64, 256, 3, 3], [64, 64, 3, 3], [128, 256, 3, 3], [128, 128, 3, 3], [128, 512, 3, 3], [128, 128, 3, 3], [256, 512, 3, 3], [256, 256, 3, 3], [256, 1024, 3, 3], [256, 256, 3, 3], [512, 1024, 3, 3], [512, 512, 3, 3], [512, 2048, 3, 3], [512, 512, 3, 3]]

layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,1536],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,1536],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                    ]
'''

'''
#resnet shape
basic_layer=[['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.2.conv1.weight', 'layer2.2.conv2.weight', 'layer2.3.conv1.weight', 'layer2.3.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.0.conv3.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.1.conv3.weight', 'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer1.2.conv3.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.conv3.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.1.conv3.weight', 'layer2.2.conv1.weight']]
global_shape=[[64, 3, 7, 7],[64, 64, 3, 3],[64, 64, 3, 3],[256, 64, 3, 3],[64, 256, 3, 3],[128, 64, 3, 3],[256, 128, 3, 3],[128, 256, 3, 3],[128, 128, 3, 3],[256, 128, 3, 3],[256, 256, 3, 3],[256, 256, 3, 3],[512, 256, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3]]
layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,192,0,0],[0,0,0,192],[0,0,0,0],[0,128,0,0],[0,0,0,128],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,256,0,0],[0,0,0,256],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,192,0,0],[0,0,0,192],[0,64,0,0],[0,192,0,0],[0,0,0,192],[0,0,0,0],[0,128,0,0],[0,128,0,128],[0,128,0,128],[0,384,0,128],[0,384,0,384],[0,384,0,384],[0,256,0,384],[0,256,0,256]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,64,0,0],[0,0,0,64],[0,64,0,0],[0,64,0,64],[0,0,0,64],[0,128,0,0],[0,128,0,128],[0,0,0,128],[0,256,0,0],[0,256,0,256],[0,0,0,256],[0,256,0,0]]
                    ]
'''

'''
#vgg
basic_layer=[['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight'],
             ['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight'],
             ['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight']]
global_shape=[[64, 3, 3, 3],[64, 64, 3, 3],[128, 64, 3, 3],[128, 128, 3, 3],[256, 128, 3, 3],[256, 256, 3, 3]]
layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                    ]
'''
transform = transforms.Compose([
         # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
         transforms.RandomCrop(32, padding=5),
         # 按0.5的概率水平翻转图片
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_SVHN = transforms.Compose([
         # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
         transforms.RandomCrop(32, padding=4),
         # 按0.5的概率水平翻转图片
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_SVHN_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

if args.trainingdata=='cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
elif args.trainingdata=='cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data100', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data100', train=False,download=True, transform=transform_test)
elif args.trainingdata=='SVHN':
    trainset = torchvision.datasets.SVHN(root='./dataSVHN', download=True, transform=transform_SVHN, split='train')
    testset = torchvision.datasets.SVHN(root='./dataSVHN', download=True, transform=transform_SVHN_test, split='test')


     
criterion = nn.CrossEntropyLoss()

if args.pretrained==1:
    pretrained=True
elif args.pretrained==0:
    pretrained=False

device1 = torch.device(gpu1 if torch.cuda.is_available() else "cpu")
device2 = torch.device(gpu2 if torch.cuda.is_available() else "cpu")
print(device1,device2)
'''device3 = torch.device(gpu3 if torch.cuda.is_available() else "cpu")
device4 = torch.device(gpu4 if torch.cuda.is_available() else "cpu")'''

loss_list=[]
a_all=[]
b_all=[]
c_all=[]
a_noniid_all=[]
b_noniid_all=[]
c_noniid_all=[]
avg_all=[]
Net_type=[]
Net=[]
Op=[]
device=[]
for _ in range(5):
    device.append(device1)
for _ in range(5):
    device.append(device2)
'''
for _ in range(25):
    device.append(device3)
for _ in range(25):
    device.append(device4)'''


for _ in range(0):
    print('resnet')
    Net_type.append(0)
    Net.append(models.resnet18(pretrained= pretrained))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.0001))
for _ in range(int(args.distribution[0])):
    print('resnet34')
    Net_type.append(1)
    Net.append(models.resnet34(pretrained= pretrained,num_classes=10))#,num_classes=10))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate[0],weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate[0],weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate[0],momentum=0.9,weight_decay=0.0001))
for _ in range(0):
    print('resnet50')
    Net_type.append(2)
    Net.append(models.resnet50(pretrained= pretrained))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))
for _ in range(int(args.distribution[1])):
    print('densenet121')
    Net_type.append(0)
    #Net.append(models.vgg13(pretrained= pretrained))
    Net.append(models.densenet121(pretrained= pretrained,num_classes=10))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate[1],weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate[1],weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate[1],momentum=0.9,weight_decay=0.001))
for _ in range(0):
    Net_type.append(1)
    Net.append(models.vgg16(pretrained= pretrained))

    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))
for _ in range(0):
    print('vgg19')
    Net_type.append(2)
    Net.append(models.vgg19(pretrained= pretrained))

    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))

print(Net_type)

node_num=len(Net_type)

print(len(trainset))
data_index=list(range(len(trainset)))
data_split_len=len(data_index)/node_num
print(int(data_split_len))
data_subset=[]
if args.iid==1:
    for i in range(node_num-1):
        data_subset.append(torch.utils.data.Subset(trainset, data_index[i*int(data_split_len):(i+1)*int(data_split_len)]))
    data_subset.append(torch.utils.data.Subset(trainset, data_index[(i+1)*int(data_split_len):]))
    random.shuffle(data_subset)
elif args.iid==0:
   
    classes=[[],[],[],[],[],[],[],[],[],[]]
    for sample in trainset:
        classes[sample[1]].extend([(sample[0],sample[1])])
    temp_classes1=list(range(node_num))
    temp_classes2=list(range(node_num))
    c1_list=[]
    c2_list=[]
    
    for node in range(node_num):
        c1=random.randint(0,len(temp_classes1)-1)
        temp_subset=classes[temp_classes1[c1]][0:int(len(classes[temp_classes1[c1]])/2)]
        c2=random.randint(0,len(temp_classes2)-1)
        while len(temp_classes2)>1 and temp_classes2[c2]==temp_classes1[c1]:
            c2=random.randint(0,len(temp_classes2)-1)
            print(c2)
        c1_list.append(temp_classes1[c1])
        c2_list.append(temp_classes2[c2])
        temp_subset.extend(classes[temp_classes2[c2]][int(len(classes[temp_classes2[c2]])/2):])
        
        data_subset.append(torch.utils.data.Subset(temp_subset, data_index[:len(temp_subset)]))
        temp_classes1.pop(c1)
        temp_classes2.pop(c2)
        print(len(data_subset[-1]))
    print(c1_list,c2_list)
    
    #########    test      ##########
    data_subset_test=[]
    test_classes=[[],[],[],[],[],[],[],[],[],[]]
    for sample in testset:
        test_classes[sample[1]].extend([(sample[0],sample[1])])
    for node in range(node_num):
        temp_subset=test_classes[c1_list[node]][0:int(len(test_classes[c1_list[node]])/2)]
        temp_subset.extend(test_classes[c2_list[node]][int(len(test_classes[c1_list[node]])/2):])
        data_subset_test.append(torch.utils.data.Subset(temp_subset, data_index[:len(temp_subset)]))




if(args.fusion==1):
    basic_layer=[]
    
    for i in range(node_num):
        
        if(args.mode=='origin'):
            layers=[]
            for key in Net[i].state_dict().keys():
                if('conv' in key):# or ('feature' in key and 'weight' in key)):
                    layers.append(key)
            temp_index=list(range(len(layers)))
            random.shuffle(temp_index)
        
            if(i < args.half_num):
                temp_index=temp_index[:int(len(layers)*args.outerratio)]
                temp_index.sort()
        
            layers_final=[]
            for index in temp_index:
                layers_final.append(layers[index])
            basic_layer.append(layers_final)
        elif (args.mode=='flop' ):
            layers=[]
            for key in Net[i].state_dict().keys():
                if('classifier' not in key) and ('fc' not in key):# or ('feature' in key and 'weight' in key)):
                    layers.append(key)
            basic_layer.append(layers)
        elif  args.mode=='hermes':
            layers=[]
            for key in Net[i].state_dict().keys():
                if('conv' in key or 'fc' in key or 'classifier' in key):# or ('feature' in key and 'weight' in key)):
                    layers.append(key)
            basic_layer.append(layers)
        elif (args.mode=='fedavg' ):
            layers=[]
            for key in Net[i].state_dict().keys():
                layers.append(key)
            basic_layer.append(layers)

        
        
        
    global_shape=global_shape_generator()
    layer_offset_range=layer_offset_range_generator()
    layer_offset=layoffset_generator()
    local_conv_index=[]
    
    global_model=[]
    for shape in global_shape:
        global_model.append(torch.randn(shape))
    
    for j in range(len(Net_type)):
        net_params = Net[j].state_dict()
        layer_shape={}
        i=0
        for key in net_params.keys():
            if key in basic_layer[j]:
                shape=list(net_params[key].shape)
                if(len(shape)==4):
                    layer_shape[key]=shape[0]*shape[1]*shape[2]*shape[3]
                elif(len(shape)==2):
                    layer_shape[key]=shape[0]*shape[1]
                elif(len(shape)==1):
                    layer_shape[key]=shape[0]
                else:
                    layer_shape[key]=1
                #layer_shape[key]=(Net_type[j]+1)*3
                '''
                layer_shape[key]=i+1
                '''
            i+=1
        local_conv_index.append(layer_shape)

    global_repeat=[]
    for shape in global_shape:
        global_repeat.append(torch.zeros(shape))
    for net_index in range(len(Net_type)):
        
        net_params = Net[net_index].state_dict()
        for key,index in zip(basic_layer[net_index],range(len(global_repeat))):
            onee=torch.ones(net_params[key].size())
            shape=list(net_params[key].size())
            
            if(len(shape)==4):
                global_repeat[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]+=onee*local_conv_index[net_index][key]
            else:
                global_repeat[index]+=onee*local_conv_index[net_index][key]
            
    masks=list(range(len(Net_type)))
    for epoch in range(args.epoch):
        start_time=time.time()
        '''
        if (epoch>200):
            for p in Op[net_index].param_groups:
                p['lr'] = 0.0001
        elif (epoch>100):
            for p in Op[net_index].param_groups:
                p['lr'] = 0.0005'''
        running_loss = 0.
        batch_size = 256
        
        add_global=[]
        for shape in global_shape:
            add_global.append(torch.zeros(shape))
    
        acc_0=[]
        acc_1=[]
        acc_2=[]
        acc_noniid_0=[]
        acc_noniid_1=[]
        acc_noniid_2=[]
        loss_all=0.0
        
        for net_index in range(int(len(Net_type))):

            if(epoch>args.startepoch+1):
                net_params = Net[net_index].state_dict()
                params = compression(net_params,net_index)
                if args.type=='inner' or args.mode=='hermes':
                    for key in basic_layer[net_index]:
                        new_param=params[key].masked_fill(~masks[net_index][key], value=0.0)
                        old_param=net_params[key].masked_fill(masks[net_index][key], value=0.0)
                        params[key]=new_param+old_param
                        kkkk=new_param==params[key]

                Net[net_index].load_state_dict(params)

                
            Net[net_index].to(device[net_index])
            for _ in range(args.uploadinterval):
                for i, data in enumerate(torch.utils.data.DataLoader(data_subset[net_index], batch_size=batch_size,shuffle=True), 0):
                        
                    inputs, labels = data
                    inputs, labels = inputs.to(device[net_index]), labels.to(device[net_index])
                    Op[net_index].zero_grad()
                    outputs = Net[net_index](inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    Op[net_index].step()
              
                loss_all+=loss.item()/len(Net_type)

            Net[net_index].to("cpu")
            
            print(net_index,loss.item())
            

            if(epoch>args.startepoch):
                net_params = Net[net_index].state_dict()
                if args.type=='inner' :
                    node_mask={}
                    for key in basic_layer[net_index]:
                        temp_neuron_list=list(range(net_params[key].shape[0]))
                        random.shuffle(temp_neuron_list)
                        temp_mask= torch.ones(net_params[key].shape).bool()
                        neuron_list=temp_neuron_list[:int(net_params[key].shape[0]*args.innerratio)]
                        neuron_list.sort()
                        temp_mask[neuron_list,:,:,:]=False
                        node_mask[key]=temp_mask  
                        net_params[key]=net_params[key].masked_fill(temp_mask, value=0.0)
                    masks[net_index]=node_mask
                elif args.mode=='hermes':
                    node_mask={}
                    for key in basic_layer[net_index]:
                        temp_neuron_list=list(range(net_params[key].shape[0]))
                        random.shuffle(temp_neuron_list)
                        temp_mask= torch.ones(net_params[key].shape).bool()
                        neuron_list=temp_neuron_list[:int(net_params[key].shape[0]*args.innerratio)]
                        neuron_list.sort()
                        if('classifier' not in key) and ('fc' not in key):
                            temp_mask[neuron_list,:,:,:]=False
                        else:
                            if('weight' in key):
                                temp_mask[neuron_list,:]=False
                                
                            elif('bias' in key):
                                temp_mask[neuron_list]=False
                                
                        node_mask[key]=temp_mask  
                        net_params[key]=net_params[key].masked_fill(temp_mask, value=0.0)
                    masks[net_index]=node_mask
                add_global = expansion(add_global,net_params,basic_layer[net_index],net_index)
        loss_list.append(loss_all)
        print('[%d] loss: %.4f' %(epoch + 1, loss_all))    
        if(epoch>args.startepoch):
            for index in range(len(add_global)):
                add_global[index]/=global_repeat[index]
                global_model[index]=add_global[index]
        
        if((epoch+1)%10== 0):
            for net_index in range(len(Net_type)):
                net_params = Net[net_index].state_dict()
                acc_iid,acc_noniid=validation(testset,Net[net_index],device[net_index],net_index)
                if(Net_type[net_index]==0):
                    acc_0.append(acc_iid)
                    acc_noniid_0.append(acc_noniid)
                elif(Net_type[net_index]==1):
                    acc_1.append(acc_iid)
                    acc_noniid_1.append(acc_noniid)
                elif(Net_type[net_index]==2):
                    acc_2.append(acc_iid)
                    acc_noniid_2.append(acc_noniid)
                Net[net_index].load_state_dict(net_params)
                Net[net_index]=Net[net_index].train()
            
            if(len(acc_0)>0):
                a_all.append(sum(acc_0)/len(acc_0))
                a_noniid_all.append(sum(acc_noniid_0)/len(acc_noniid_0))
                print('net_index_A:Accuracy',net_index, sum(acc_0)/len(acc_0),sum(acc_noniid_0)/len(acc_noniid_0))
            if(len(acc_1)>0):
                b_all.append(sum(acc_1)/len(acc_1))
                b_noniid_all.append(sum(acc_noniid_1)/len(acc_noniid_1))
                print('net_index_B:Accuracy',net_index, sum(acc_1)/len(acc_1),sum(acc_noniid_1)/len(acc_noniid_1))
            if(len(acc_2)>0):
                c_all.append(sum(acc_2)/len(acc_2))
                c_noniid_all.append(sum(acc_noniid_2)/len(acc_noniid_2))
                print('net_index_C:Accuracy',net_index, sum(acc_2)/len(acc_2),sum(acc_noniid_2)/len(acc_noniid_2))
        end_time=time.time()
        
        print('duration time',end_time-start_time)
    print(loss_list)
    print(a_all)
    print(b_all)
    print(c_all)
    print(a_noniid_all)
    print(b_noniid_all)
    print(c_noniid_all)
    for ac in range(len(a_all)):
        avg_all.append((a_all[ac]+b_all[ac]+c_all[ac])/3)
    print(avg_all)
    print('Finished Training')
